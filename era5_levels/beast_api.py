"""Single adapter for every `beast` touch-point.

`beast` is installed separately (``pip install -e .`` in your beast checkout)
and is mid-rename: its internal modules currently import under the ``gb.*``
namespace while the package is moving to ``beast.*``. Rather than scatter that
uncertainty across the codebase, every import of beast goes through here, with a
``beast`` → ``gb`` fallback. When the rename settles you change ONE file.

Imports are done lazily inside functions so the pure-Python parts of this repo
(config derivation, variable layout, transfer logic) import fine on a laptop
with no beast / GPU / jigsaw present.
"""

from __future__ import annotations

import importlib
import os


def _imp(module_suffix: str):
    """Import ``beast.<suffix>`` falling back to ``gb.<suffix>``."""
    for root in ("beast", "gb"):
        try:
            return importlib.import_module(f"{root}.{module_suffix}")
        except ModuleNotFoundError as e:
            # only swallow "no top-level package" misses, not real errors inside
            if e.name not in ("beast", "gb"):
                raise
            last = e
    raise ModuleNotFoundError(
        f"Could not import '{module_suffix}' from either 'beast' or 'gb'. "
        f"Install beast (pip install -e .) in your beast checkout."
    ) from last


# --- model / wrappers --------------------------------------------------------
def get_model_classes():
    m = _imp("model.model_gb")
    t = _imp("model.model_gb_tiny")
    return m.Bellbeast, t.TinyBellbeast


def get_expert_class():
    return _imp("model.expert").Expert


# --- distributed comm --------------------------------------------------------
def get_comm():
    """Return (create_process_mesh_dict, get_process_group)."""
    c = _imp("comm")
    return c.create_process_mesh_dict, c.get_process_group


def get_process_group(name: str):
    _, gpg = get_comm()
    return gpg(name)


# --- data --------------------------------------------------------------------
def get_dataloader_fn():
    return _imp("data.dataloader").get_dataloader


# --- utils we reuse ----------------------------------------------------------
def get_utils():
    return _imp("utils")


# --- distributed bootstrap ---------------------------------------------------
def bootstrap_distributed(mesh_dims):
    """Initialise torch.distributed from the SLURM/env vars, pin the GPU, and
    build beast's process-mesh dictionary.

    beast.comm.create_process_mesh_dict requires torch.distributed to already be
    initialised (it raises otherwise), so we do that here from the standard
    launch-environment variables. Works under `srun` (SLURM_* vars) and under
    `torchrun` (RANK/WORLD_SIZE/LOCAL_RANK).
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
