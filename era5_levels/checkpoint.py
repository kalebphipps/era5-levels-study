"""Brief checkpoint before beast checkpoint functionality is implemented."""

from __future__ import annotations

import os
import re

import torch

from . import beast_api

_CKPT_RE = re.compile(r"cp-epoch(\d+)(?:-step(\d+))?-e(\d+)-jc(\d+)\.pt$")


def save_checkpoint(checkpoint_path, model, optimizer, epoch, step, loss) -> None:
    """Write this shard's checkpoint (only DDP-rank 0 of the shard writes).

    Parameters
    ----------
    checkpoint_path : str
        Directory to write the ``.pt`` file into (created if missing).
    model : torch.nn.Module
        The (DDP-wrapped) model whose ``state_dict`` is saved.
    optimizer : torch.optim.Optimizer
        Optimizer whose ``state_dict`` is saved.
    epoch : int
        Epoch index (encoded in the filename).
    step : int
        Step index within the epoch (encoded in the filename).
    loss : float or torch.Tensor
        Loss value to store (``.item()`` is called if it is a tensor).
    """
    channel_group = beast_api.get_process_group("JChannel")
    ep_group = beast_api.get_process_group("Expert")
    ddp_rank = beast_api.get_process_group("DDP").rank()

    if ddp_rank != 0:
        return

    loss_val = loss.item() if hasattr(loss, "item") else loss
    cp = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "ep_group_rank": ep_group.rank(),
        "channel_group_rank": channel_group.rank(),
        "loss": loss_val,
        "format": 1,
    }
    name = f"cp-epoch{epoch}-step{step}-e{ep_group.rank()}-jc{channel_group.rank()}"
    os.makedirs(checkpoint_path, exist_ok=True)
    torch.save(cp, os.path.join(checkpoint_path, f"{name}.pt"))


def _load_one(path, map_location="cpu"):
    """Load a single checkpoint file.

    Parameters
    ----------
    path : str
        Path to the ``.pt`` checkpoint file.
    map_location : str or torch.device, optional
        Passed to ``torch.load`` (default ``"cpu"``).

    Returns
    -------
    tuple
        ``(model_sd, optim_sd, epoch, step, ep_rank, jc_rank, loss)``.
    """
    cp = torch.load(path, map_location=map_location, weights_only=False)
    return (
        cp["model_state_dict"],
        cp["optimizer_state_dict"],
        cp["epoch"],
        cp.get("step", None),
        cp["ep_group_rank"],
        cp["channel_group_rank"],
        cp["loss"],
    )


def load_latest_model_states(directory):
    """Return the latest checkpoint tuple for this rank.

    Parameters
    ----------
    directory : str
        The run directory.

    Returns
    -------
    tuple
        ``(model_sd, optim_sd, epoch, step, ep_rank, jc_rank, loss)`` for the
        latest matching checkpoint.

    Raises
    ------
    FileNotFoundError
        If no checkpoint exists for this rank's (ep, jc) shard.
    """
    ckpt_dir = os.path.join(directory, "checkpoints")
    ep_rank = beast_api.get_process_group("Expert").rank()
    jc_rank = beast_api.get_process_group("JChannel").rank()

    candidates = []
    for fname in os.listdir(ckpt_dir):
        m = _CKPT_RE.match(fname)
        if not m:
            continue
        epoch = int(m.group(1))
        step = int(m.group(2)) if m.group(2) else -1
        e, jc = int(m.group(3)), int(m.group(4))
        if e == ep_rank and jc == jc_rank:
            candidates.append((epoch, step, fname))

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint for ep={ep_rank}, jc={jc_rank} in {ckpt_dir}")

    epoch, step, fname = max(candidates, key=lambda x: (x[0], x[1]))
    return _load_one(os.path.join(ckpt_dir, fname))


def load_state_dict(model, optimizer, model_state_dict, optimizer_state_dict, last_epoch):
    """Load weights and optimizer into the (DDP-wrapped) model.

    Parameters
    ----------
    model : torch.nn.Module
        The (DDP-wrapped) model to load into.
    optimizer : torch.optim.Optimizer
        The optimizer to load into.
    model_state_dict : dict
        Saved model state dict.
    optimizer_state_dict : dict
        Saved optimizer state dict.
    last_epoch : int
        The epoch the checkpoint was saved at.

    Returns
    -------
    model : torch.nn.Module
        The (possibly updated) model.
    optimizer : torch.optim.Optimizer
        The (possibly updated) optimizer.
    start_epoch : int
        ``last_epoch + 1`` on success, or ``0`` if the load failed.
    """
    try:
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        start_epoch = last_epoch + 1
    except Exception as err:  # noqa: BLE001 - resume is best-effort
        import torch.distributed as dist

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"[resume] state_dict load failed ({err}); starting fresh")
        start_epoch = 0
    return model, optimizer, start_epoch
