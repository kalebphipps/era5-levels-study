"""Single adapter for every ``beast`` touch-point.

``beast`` is installed separately (``pip install -e .`` in your beast checkout,
plus its jigsaw submodule). Rather than scatter beast imports across the
codebase, every import of beast goes through this module. If beast moves a class
or renames a module, this is the ONE file to change.

Imports are done lazily inside functions so the pure-Python parts of this repo
(config derivation, variable layout, transfer logic) import fine on a laptop
with no beast / GPU / jigsaw present.
"""

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
            raise  # a real import error inside beast, not a missing-package miss
        raise ModuleNotFoundError(
            f"Could not import 'beast.{module_suffix}'. Install beast "
            "(pip install -e .) in your beast checkout, plus its jigsaw submodule."
        ) from err


# --- model / wrappers --------------------------------------------------------
def get_model_class() -> type:
    """Return the unified ``beast.model.Beast`` model class.

    The old ``Bellbeast`` / ``TinyBellbeast`` split is gone after the model
    refactor: a single ``Beast`` now handles both the single-GPU and
    domain-parallel cases (process groups of size 1 vs >1). It is built via its
    keyword-only constructor in :func:`era5_levels.train.build_model` (a
    config-driven factory ``beast.model.get_model`` also exists if you'd rather
    use named presets).

    Returns
    -------
    type
        The ``beast.model.Beast`` class.
    """
    return _imp("model").Beast


def get_expert_class() -> type:
    """Return the ``beast.layers.expert.Expert`` wrapper class.

    Returns
    -------
    type
        The ``Expert`` class (relocated by the refactor from
        ``beast.model.expert`` to ``beast.layers.expert``).
    """
    return _imp("layers.expert").Expert


# --- distributed comm --------------------------------------------------------
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


# --- data --------------------------------------------------------------------
def get_dataloader_fn() -> Any:
    """Return ``beast.data.dataloader.get_dataloader``.

    Returns
    -------
    callable
        The dataloader factory used by the training loop.
    """
    return _imp("data.dataloader").get_dataloader


# --- evaluation --------------------------------------------------------------
def get_evaluation() -> ModuleType:
    """Return beast's ``evaluation`` subpackage.

    Exposes the house-standard, tested, sharding-aware metric helpers
    (``mse``, ``rmse``, ``latitude_weighted_average``, ``select_variables``,
    ``make_variable_names``, ``gather_along_dimension``) that the study's
    ``evaluate`` module orchestrates rather than reimplementing.

    Returns
    -------
    ModuleType
        The ``beast.evaluation`` package.
    """
    return _imp("evaluation")


# --- utils we reuse ----------------------------------------------------------
def get_utils() -> ModuleType:
    """Return the ``beast.utils`` module (seeding, dtype, spatial weights).

    Returns
    -------
    ModuleType
        The ``beast.utils`` module.
    """
    return _imp("utils")


# --- distributed bootstrap ---------------------------------------------------
def bootstrap_distributed(mesh_dims: list[int]) -> tuple[dict, int, int, int]:
    """Initialise ``torch.distributed``, pin the GPU, and build the process mesh.

    ``beast.comm.create_process_mesh_dict`` requires ``torch.distributed`` to be
    already initialised (it raises otherwise), so this initialises it from the
    standard launch-environment variables. Works under ``srun`` (``SLURM_*``
    vars) and under ``torchrun`` (``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK``).

    Parameters
    ----------
    mesh_dims : list of int
        The five mesh dimensions ``[experts, dp, up, jspatial, jchannel]``; the
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
