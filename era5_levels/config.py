"""Configuration loading + channel derivation for the levels study.

The whole point of the study is that switching 13 ↔ 37 levels is a *one-field*
change (`data.pressure_levels`). This module derives every channel count from
that field via `variable_layout.num_variables`, so the two runs share an
otherwise byte-identical config and the comparison stays fair.

It is deliberately self-contained (does not import beast's own config, which is
in flux and not level-generalized).
"""

from __future__ import annotations

import os

import yaml

from .variable_layout import num_variables


def load_config(path: str, overlays: list[str] | None = None) -> dict:
    """Load a base YAML and apply optional overlay YAMLs (later wins).

    Overlays let you keep one shared `base_0p25.yaml` and tiny `levels13.yaml` /
    `levels37.yaml` files that only set `data.pressure_levels`.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ov in overlays or []:
        with open(ov) as f:
            _deep_update(cfg, yaml.safe_load(f) or {})
    return cfg


def _deep_update(base: dict, new: dict) -> dict:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def finalize_config(cfg: dict) -> dict:
    """Derive channel counts + resolve env-dependent paths/mesh.

    Mirrors the essential parts of beast's setup_config_dict, but drives the
    channel counts from `pressure_levels` so 13 vs 37 is automatic.
    """
    data, model = cfg["data"], cfg["model"]

    # --- channel counts from the level set (the fair-comparison knob) --------
    if "pressure_levels" not in data:
        raise KeyError("data.pressure_levels is required (the 13- or 37-level list).")
    n_vars = num_variables(data["pressure_levels"])
    if data.get("n_variables") not in (None, n_vars):
        print(f"[config] overriding data.n_variables "
              f"({data.get('n_variables')}) -> {n_vars} (from pressure_levels)")
    data["n_variables"] = n_vars
    # Output channels = predicted variables x output timesteps (the dataset
    # concatenates target timesteps along the channel axis).
    n_out = data.get("n_out_timesteps", 1)
    model["n_output_channels"] = n_out * n_vars

    # constant masks: padded to an even count when model-parallel (matches the
    # dataset, which pads so the channel split stays balanced)
    n_masks = len(data.get("constant_masks", []))
    if cfg.get("jigsaw", {}).get("parallelism", 1) > 1 and n_masks % 2 == 1:
        n_masks += 1

    n_in = data.get("n_in_timesteps", 1)
    if n_in not in (1, 2):
        print(f"[config] n_in_timesteps={n_in} unsupported -> using 1")
        n_in = data["n_in_timesteps"] = 1
    model["n_input_channels"] = n_in * n_vars + n_masks

    # --- mesh: allow a single -1 entry to be inferred from world size --------
    world = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1)))
    mesh = cfg.get("mesh_dims")
    if mesh and -1 in mesh:
        prod = 1
        for m in mesh:
            if m != -1:
                prod *= m
        mesh[mesh.index(-1)] = max(world // prod, 1)

    # --- prepend DATA_DIR to data paths if set -------------------------------
    data_dir = os.environ.get("DATA_DIR")
    if data_dir:
        for key in ("train_data_path", "valid_data_path", "constant_masks_path"):
            if key in data and not data[key].startswith("/"):
                data[key] = os.path.join(data_dir, data[key])
    workdir = os.environ.get("WORKDIR")
    if workdir and "normalization_dir" in data and not data["normalization_dir"].startswith("/"):
        data["normalization_dir"] = os.path.join(workdir, data["normalization_dir"])

    return cfg
