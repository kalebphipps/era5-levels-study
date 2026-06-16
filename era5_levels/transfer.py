"""Code to perform the frozen core transfer experiment."""

from __future__ import annotations

import torch

# Only layers dependent on levels - everything else frozen.
IO_LAYER_KEYS = ("patch_embedding", "patch_recovery")


def is_io_param(name: str) -> bool:
    """Return whether a parameter name belongs to a level-dependent I/O layer.

    Parameters
    ----------
    name : str
        A parameter name from ``model.named_parameters()``.

    Returns
    -------
    bool
        ``True`` if the name matches one of the I/O layers instead of core.
    """
    return any(k in name for k in IO_LAYER_KEYS)


def freeze_core(model: torch.nn.Module) -> tuple[int, int]:
    """Freeze every parameter except the I/O.

    Parameters
    ----------
    model : torch.nn.Module
        The model to freeze in place.

    Returns
    -------
    n_train : int
        Number of (still-trainable) I/O-conv parameters.
    n_frozen : int
        Number of frozen core parameters.
    """
    n_train = n_frozen = 0
    for name, p in model.named_parameters():
        if is_io_param(name):
            p.requires_grad_(True)
            n_train += p.numel()
        else:
            p.requires_grad_(False)
            n_frozen += p.numel()
    return n_train, n_frozen


def load_core_from_checkpoint(model: torch.nn.Module, state_dict: dict) -> list[str]:
    """Copy only the shared-core tensors from a checkpoint into ``model``.

    Parameters
    ----------
    model : torch.nn.Module
        The (freshly built) target model to load the core into, in place.
    state_dict : dict
        Source state dict from a trained checkpoint of the other level count.

    Returns
    -------
    list of str
        The keys that were actually loaded (for logging).
    """
    own = model.state_dict()
    to_load = {
        k: v for k, v in state_dict.items()
        if (not is_io_param(k)) and k in own and own[k].shape == v.shape
    }
    own.update(to_load)
    model.load_state_dict(own, strict=False)
    return list(to_load.keys())
