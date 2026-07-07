"""Telemetry for the feasibility / scaling runs."""

from __future__ import annotations

import csv
import json
import os
import time

import torch
import torch.distributed as dist


def _is_root() -> bool:
    """Return ``True`` on the single logging rank (or when not distributed)."""
    return (not dist.is_initialized()) or dist.get_rank() == 0


def _peak_mem_gb(device) -> float:
    """Return the peak allocated GPU memory in GB (0.0 if CUDA is unavailable).

    Parameters
    ----------
    device : torch.device
        Device to query.

    Returns
    -------
    float
        Peak allocated memory in gigabytes since the last reset.
    """
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1e9


def write_run_summary(path, cfg, model, groups, device) -> None:
    """Write ``run_summary.json`` with the static facts about this run.

    Parameters
    ----------
    path : str
        Run directory.
    cfg : dict
        The full config.
    model : torch.nn.Module
        The (already built / wrapped) model, for parameter counting.
    groups : dict
        Mapping of process-group name to group (``DP`` is used for data-parallel
        size).
    device : torch.device
        Compute device, for the GPU name.
    """
    if not _is_root():
        return
    try:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        data, mdl, tr = cfg["data"], cfg["model"], cfg["training"]
        world = dist.get_world_size() if dist.is_initialized() else 1
        summary = {
            "world_size_gpus": world,
            "mesh_dims": cfg.get("mesh_dims"),
            "dp_size": groups["DP"].size() if "DP" in groups else None,
            "resolution_lat_lon": [data.get("xlat"), data.get("xlon")],
            "n_pressure_levels": len(data.get("pressure_levels", [])),
            "n_input_channels": mdl.get("n_input_channels"),
            "n_output_channels": mdl.get("n_output_channels"),
            "embedding_dimension": mdl.get("embedding_dimension"),
            "n_attn_blocks": mdl.get("n_attn_blocks"),
            "n_attn_heads": mdl.get("n_attn_heads"),
            "patch_size": mdl.get("patch_size"),
            "batch_size_per_replica": tr.get("batch_size"),
            "dtype": tr.get("dtype"),
            "n_epochs_planned": tr.get("n_epochs"),
            "params_total": total,
            "params_total_millions": round(total / 1e6, 3),
            "params_trainable": trainable,
            "gpu": (torch.cuda.get_device_name(device)
                    if torch.cuda.is_available() else "cpu"),
        }
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "run_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
            f.flush()
        print(f"[telemetry] run_summary.json  params={total / 1e6:.1f}M  "
              f"world={world}  res={data.get('xlat')}x{data.get('xlon')}  "
              f"levels={len(data.get('pressure_levels', []))}", flush=True)
    except Exception as e:  # noqa: BLE001 - telemetry must never crash training
        if _is_root():
            print(f"[telemetry] run_summary failed: {e}", flush=True)


def log_epoch(path, epoch, n_steps, epoch_seconds, global_batch, peak_mem_gb,
              train_loss, valid_loss=None) -> None:
    """Append one per-epoch row to ``throughput.csv``.

    Parameters
    ----------
    path : str
        Run directory.
    epoch : int
        Epoch index just completed.
    n_steps : int
        Number of optimizer steps in the epoch.
    epoch_seconds : float
        Wall-clock duration of the training epoch.
    global_batch : int
        Samples processed globally per step (``batch_size * dp_size``).
    peak_mem_gb : float
        Peak GPU memory during the epoch, in GB.
    train_loss : float
        Mean training loss.
    valid_loss : float, optional
        Mean validation loss, if computed.
    """
    if not _is_root():
        return
    try:
        step_s = epoch_seconds / max(1, n_steps)
        samples_s = (n_steps * global_batch) / max(1e-9, epoch_seconds)
        row = {
            "epoch": epoch,
            "n_steps": n_steps,
            "epoch_seconds": round(epoch_seconds, 2),
            "mean_step_seconds": round(step_s, 4),
            "global_samples_per_second": round(samples_s, 3),
            "peak_mem_gb": round(peak_mem_gb, 2),
            "train_loss": round(float(train_loss), 6),
            "valid_loss": (None if valid_loss is None
                           else round(float(valid_loss), 6)),
        }
        fpath = os.path.join(path, "throughput.csv")
        new = not os.path.exists(fpath)
        with open(fpath, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
            f.flush()
        print(f"[telemetry] epoch {epoch}: {step_s:.3f}s/step  "
              f"{samples_s:.2f} samples/s  peak {peak_mem_gb:.1f}GB", flush=True)
    except Exception as e:  # noqa: BLE001
        if _is_root():
            print(f"[telemetry] log_epoch failed: {e}", flush=True)


class StepThroughputLogger:
    """Append windowed throughput.

    Parameters
    ----------
    path : str
        Run directory.
    global_batch : int
        Samples processed globally per step (``batch_size * dp_size``).
    device : torch.device
        Device, for the peak-memory query.
    """

    def __init__(self, path, global_batch, device):
        self.fpath = os.path.join(path, "step_timings.csv")
        self.global_batch = global_batch
        self.device = device
        self._t = None
        self._step = None
        self._need_header = not os.path.exists(self.fpath)

    def tick(self, epoch, step) -> None:
        """Record the window since the previous tick.

        Parameters
        ----------
        epoch : int
            Current epoch index.
        step : int
            Current step index within the epoch.
        """
        if not _is_root():
            return
        try:
            now = time.perf_counter()
            if self._t is not None and step > self._step:
                dt = now - self._t
                nst = step - self._step
                row = {
                    "epoch": epoch,
                    "step": step,
                    "window_steps": nst,
                    "window_seconds": round(dt, 3),
                    "sec_per_step": round(dt / nst, 4),
                    "global_samples_per_second": round(
                        nst * self.global_batch / max(1e-9, dt), 3),
                    "peak_mem_gb": round(_peak_mem_gb(self.device), 2),
                }
                with open(self.fpath, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(row))
                    if self._need_header:
                        w.writeheader()
                        self._need_header = False
                    w.writerow(row)
                    f.flush()
            self._t = now
            self._step = step
        except Exception as e:  # noqa: BLE001
            if _is_root():
                print(f"[telemetry] step tick failed: {e}", flush=True)
