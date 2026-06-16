"""Config loading to simplify different runs for comparison."""

from __future__ import annotations

import os

import yaml

from .variable_layout import num_variables


def load_config(path: str, overlays: list[str] | None = None) -> dict:
    """Load a base YAML and apply optional overlay YAMLs.

    Overlays add on additional configs on overwrite options defined in ``base_*.yaml`` - with multiple overlays the
    latest overlay always overwrites previous overlays (so order is important).

    Parameters
    ----------
    path : str
        Path to the base YAML config.
    overlays : list of str, optional
        Paths to overlay YAMLs, applied in order on top of the base. Later
        overlays override earlier ones and the base (deep merge).

    Returns
    -------
    dict
        The merged configuration dictionary.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ov in overlays or []:
        with open(ov) as f:
            _deep_update(cfg, yaml.safe_load(f) or {})
    return cfg


def _deep_update(base: dict, new: dict) -> dict:
    """Recursively update new configs into base config.

    Parameters
    ----------
    base : dict
        Dictionary to update in place.
    new : dict
        Dictionary whose entries override ``base``.

    Returns
    -------
    dict
        The mutated ``base`` dictionary.
    """
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def finalize_config(cfg: dict) -> dict:
    """Derive channel count for different pressure levels whilst ensuring the rest of the config remains identical.

    Parameters
    ----------
    cfg : dict
        The loaded (pre-finalized) configuration dictionary.

    Returns
    -------
    dict
        The same dictionary, with derived ``n_variables`` / ``n_input_channels`` / ``n_output_channels``, the
        resolved mesh, and absolute data paths.

    Raises
    ------
    KeyError
        If ``data.pressure_levels`` is missing.
    """
    data, model = cfg["data"], cfg["model"]

    if "pressure_levels" not in data:
        raise KeyError("data.pressure_levels is required (the 13- or 37-level list).")
    n_vars = num_variables(data["pressure_levels"])
    if data.get("n_variables") not in (None, n_vars):
        print(f"[config] overriding data.n_variables "
              f"({data.get('n_variables')}) -> {n_vars} (from pressure_levels)")
    data["n_variables"] = n_vars
    n_out = data.get("n_out_timesteps", 1)
    model["n_output_channels"] = n_out * n_vars

    n_masks = len(data.get("constant_masks", []))
    if cfg.get("jigsaw", {}).get("parallelism", 1) > 1 and n_masks % 2 == 1:
        n_masks += 1

    n_in = data.get("n_in_timesteps", 1)
    if n_in not in (1, 2):
        print(f"[config] n_in_timesteps={n_in} unsupported -> using 1")
        n_in = data["n_in_timesteps"] = 1
    model["n_input_channels"] = n_in * n_vars + n_masks

    world = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1)))
    mesh = cfg.get("mesh_dims")
    if mesh and -1 in mesh:
        prod = 1
        for m in mesh:
            if m != -1:
                prod *= m
        mesh[mesh.index(-1)] = max(world // prod, 1)

    # Beast on cluster still uses multiple data paths, this is redundant.
    if "data_path" in data:
        data["train_data_path"] = data["valid_data_path"] = data["data_path"]

    data_dir = os.environ.get("DATA_DIR")
    if data_dir:
        for key in ("data_path", "train_data_path", "valid_data_path",
                    "constant_masks_path"):
            if key in data and not data[key].startswith("/"):
                data[key] = os.path.join(data_dir, data[key])
    workdir = os.environ.get("WORKDIR")
    if workdir and "normalization_dir" in data and not data["normalization_dir"].startswith("/"):
        data["normalization_dir"] = os.path.join(workdir, data["normalization_dir"])

    return cfg
