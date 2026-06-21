"""Evaluate on only a subset of the model."""

from __future__ import annotations

import argparse
import csv
import os

import torch

from era5_levels.variable_layout import PRESSURE_LEVELS_13, PRESSURE_LEVELS_37

DEFAULT_MAP_VARS = ["geopotential_500", "2m_temperature", "temperature_850"]


def check_indices():
    """Print where the 13 standard levels land inside the 37-level layout."""
    from era5_levels.evaluate import subset_indices
    idx = subset_indices(PRESSURE_LEVELS_37, PRESSURE_LEVELS_13)
    print(f"37-level layout has {6 + 6 * len(PRESSURE_LEVELS_37)} channels; "
          f"13-level subset selects {len(idx)} of them.")
    print("first/last subset channel indices:", idx[:6].tolist(), "...", idx[-3:].tolist())


def run_distributed(args):
    """Rebuild the sharded model, load its checkpoint, and score the 13-level subset.

    Runs inside the same process mesh as training (one task per GPU). Per-variable
    RMSE is reduced across the mesh and restricted to the 13 standard levels, then
    written to ``<results-dir>/subset_metrics.csv`` on rank 0.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments (``config``, ``overlay``, ``results_dir``, ...).
    """
    import torch.distributed as dist

    from era5_levels import beast_api, checkpoint
    from era5_levels.config import finalize_config, load_config
    from era5_levels.evaluate import dump_sample_maps, validate_per_level
    from era5_levels.train import build_model

    cfg = finalize_config(load_config(args.config, args.overlay))
    beast_api.bootstrap_distributed(cfg["mesh_dims"])
    _, get_pg = beast_api.get_comm()
    groups = {n: get_pg(n) for n in
              ("JSpatial", "JChannel", "DTP", "SP", "DP", "DDP", "Expert")}
    utils = beast_api.get_utils()
    Expert = beast_api.get_expert_class()
    get_dataloader = beast_api.get_dataloader_fn()

    dtype = utils.set_dtype(cfg["training"]["dtype"])
    device = torch.device("cuda", torch.cuda.current_device())

    model = build_model(cfg, device, dtype, groups)
    model = Expert(model, groups["Expert"])

    # load this rank's checkpoint shard (saved per ep/jchannel rank). Use the
    # vendored helper — beast.utils no longer provides load_latest_model_states
    # (the sharded save/resume helpers live in era5_levels.checkpoint).
    state = checkpoint.load_latest_model_states(args.results_dir)[0]
    # strip a possible DDP "module." prefix
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=False)

    valid_dl = get_dataloader(cfg, groups["DTP"].rank(), groups["DTP"].size(),
                              groups["DP"].rank(), groups["DP"].size(),
                              mode="validation", dtype=dtype)

    # per-variable std (physical units) if available on the dataset. The
    # dataloader's dataset may be wrapped in a torch Subset (valid_subset > 0),
    # which hides norm_values — unwrap nested .dataset attrs until we reach the
    # underlying ERA5 dataset that carries them. Lets physical-unit RMSE work
    # when scoring a checkpoint on a subset (e.g. a mid-training spot check).
    ds = getattr(valid_dl, "dataset", None)
    while ds is not None and not hasattr(ds, "norm_values"):
        ds = getattr(ds, "dataset", None)
    data_std = getattr(ds, "norm_values", {}).get("std") if ds is not None else None

    names, rmse = validate_per_level(
        model, valid_dl, cfg["data"]["pressure_levels"], groups, device,
        subset_levels=PRESSURE_LEVELS_13, data_std=data_std,
    )

    # Maps.
    if args.dump_maps:
        dump_sample_maps(
            model, valid_dl, cfg["data"]["pressure_levels"], groups, device,
            args.map_vars, os.path.join(args.results_dir, "maps"),
        )

    if dist.get_rank() == 0:
        out_csv = args.out_csv or os.path.join(args.results_dir, "subset_metrics.csv")
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["variable", "rmse"])
            for n, v in zip(names, rmse.tolist()):
                w.writerow([n, f"{v:.6f}"])
        print(f"{'variable':32s} {'rmse':>12s}")
        for n, v in zip(names, rmse.tolist()):
            print(f"{n:32s} {v:12.4f}")
        print(f"\nmean RMSE over 13 common levels: {rmse.mean().item():.4f}")
        print(f"wrote {out_csv}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-indices", action="store_true",
                    help="pure-Python channel-index sanity check (no GPU/beast)")
    ap.add_argument("--config")
    ap.add_argument("--overlay", action="append", default=[])
    ap.add_argument("--results-dir", help="run dir containing checkpoints/ to evaluate")
    ap.add_argument("--out-csv", help="where to write the per-variable RMSE "
                    "(default <results-dir>/subset_metrics.csv)")
    ap.add_argument("--dump-maps", action="store_true",
                    help="also dump <var>_{pred,true,err}.npy maps for the poster")
    ap.add_argument("--map-vars", nargs="*", default=DEFAULT_MAP_VARS,
                    help="variables to dump maps for (with --dump-maps)")
    args = ap.parse_args()

    if args.check_indices:
        check_indices()
    elif args.config and args.results_dir:
        run_distributed(args)
    else:
        ap.error("give --check-indices, or --config + --results-dir for the "
                 "distributed evaluation.")
