"""Wrapper to deal with beast imports easily whilst code base is changing."""

from __future__ import annotations

import importlib
import os
from types import ModuleType
from typing import Any


def _imp(module_suffix: str) -> ModuleType:
    """Import a submodule of the ``beast`` package.

    Parameters
    ----------
    module_suffix : str
        Dotted path relative to the ``beast`` package, e.g. ``"data.dataloader"``.

    Returns
    -------
    ModuleType
        The imported ``beast.<module_suffix>`` module.

    Raises
    ------
    ModuleNotFoundError
        If the top-level ``beast`` package is not importable (i.e. not
        installed). Errors raised *inside* a found module are propagated
        unchanged.
    """
    try:
        return importlib.import_module(f"beast.{module_suffix}")
    except ModuleNotFoundError as err:
        if err.name != "beast":
            raise
        raise ModuleNotFoundError(
            f"Could not import 'beast.{module_suffix}'. Install beast "
            "(pip install -e .) in your beast checkout, plus its jigsaw submodule."
        ) from err


def get_model_class() -> type:
    """Return the unified Beast model class.

    Returns
    -------
    type
        The ``beast.model.Beast`` class.
    """
    return _imp("model").Beast


def get_expert_class() -> type:
    """Return the Expert wrapper class.

    Returns
    -------
    type
        The ``Expert`` class.
    """
    return _imp("layers.expert").Expert


def get_comm() -> tuple[Any, Any]:
    """Return beast's process-mesh helpers.

    Returns
    -------
    tuple of callable
        ``(create_process_mesh_dict, get_process_group)`` from ``beast.comm``.
    """
    c = _imp("comm")
    return c.create_process_mesh_dict, c.get_process_group


def get_process_group(name: str) -> Any:
    """Return a previously created process group by name.

    Parameters
    ----------
    name : str
        Logical group name, e.g. ``"JChannel"``, ``"JSpatial"``, ``"DDP"``.

    Returns
    -------
    torch.distributed.ProcessGroup
        The requested process group.
    """
    _, gpg = get_comm()
    return gpg(name)

def get_dataloader_fn() -> Any:
    """Return a dataloader.

    Returns
    -------
    callable
        The dataloader used by the training loop.
    """
    return _imp("data.dataloader").get_dataloader


def get_evaluation() -> ModuleType:
    """Return evaluation subpackage.

    Returns
    -------
    ModuleType
        The ``beast.evaluation`` package.
    """
    return _imp("evaluation")


def get_utils() -> ModuleType:
    """Return the utils module.

    Returns
    -------
    ModuleType
        The ``beast.utils`` module.
    """
    return _imp("utils")


def bootstrap_distributed(mesh_dims: list[int]) -> tuple[dict, int, int, int]:
    """Set up distribution and process mesh.

    Parameters
    ----------
    mesh_dims : list of int
        The five mesh dimensions ``[experts, dp, up, jspatial, jchannel]`` where the
        product must equal the world size.

    Returns
    -------
    pg_dict : dict
        Mapping of logical group name to ``torch.distributed.ProcessGroup``.
    rank : int
        Global rank of this process.
    world : int
        Total number of processes (world size).
    local_rank : int
        Node-local rank, used to pin the CUDA device.
    """
    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    local_rank = int(
        os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", rank))
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world)

    create_process_mesh_dict, _ = get_comm()
    pg_dict = create_process_mesh_dict(mesh_dims)
    return pg_dict, rank, world, local_rank
