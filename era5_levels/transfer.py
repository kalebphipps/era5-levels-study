"""Frozen-core transfer experiment.

Because pressure levels live on the channel axis, only two layers of the model
depend on the number of levels: ``patch_embedding`` (in_channels -> embed) and
``patch_recovery`` (embed -> out_channels). The entire SWIN core
(downsample / transformer blocks / upsample / mix) runs in embed-dim space and
is architecturally identical between the 13- and 37-level models.

That makes a cheap, revealing ablation possible: take a trained core, FREEZE it,
and retrain ONLY the two thin I/O convs for the *other* level count. If the
frozen core reaches (nearly) the same skill as the fully-trained model, the
benefit of extra levels lives in the *representation*; if it can't, the benefit
is bottlenecked at the I/O. Few params train, so it's fast.

Usage in the training entrypoint: after building the (fresh) model for the
target level count, load the source checkpoint into the core, then call
``freeze_core(model)`` before constructing the optimizer.
"""

from __future__ import annotations

import torch

# Substrings identifying the only level-count-dependent layers. Everything else
# is the shared core. (Matches the low-LR key list in the reference loop.)
IO_LAYER_KEYS = ("patch_embedding", "patch_recovery")


def is_io_param(name: str) -> bool:
    """Return whether a parameter name belongs to a level-dependent I/O conv.

    Parameters
    ----------
    name : str
        A (dotted) parameter name from ``model.named_parameters()``.

    Returns
    -------
    bool
        ``True`` if the name matches one of :data:`IO_LAYER_KEYS`
        (``patch_embedding`` / ``patch_recovery``), i.e. a level-count-dependent
        I/O layer rather than the shared core.
    """
    return any(k in name for k in IO_LAYER_KEYS)


def freeze_core(model: torch.nn.Module) -> tuple[int, int]:
    """Freeze every parameter except the I/O convs.

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

    I/O-conv tensors are skipped (their shapes differ between level counts), as
    are any keys whose shape does not match the target model. Uses
    ``strict=False`` semantics: missing or mismatched keys are tolerated.

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
