"""Entrypoint: bootstrap distributed + mesh, then run the deterministic loop.

Launched once per GPU under `srun` (or `torchrun`):

    python -m era5_levels.main --config configs/base_0p25.yaml \\
        --overlay configs/levels37.yaml

The `--overlay` files are applied on top of the base (later wins), so the only
difference between a 13- and a 37-level run is which one-line overlay you pass.
"""

from __future__ import annotations

import argparse
import os

import torch.distributed as dist

from . import beast_api
from .config import finalize_config, load_config


def resolve_run_dir(arg: str | None) -> str:
    """Resolve the stable output dir for checkpoints + metrics.

    Chained jobs share the same RUN_DIR so a job that resumes finds the previous
    job's checkpoints (the per-jobid fallback is only for one-off interactive
    runs, which don't resume).

    Parameters
    ----------
    arg : str or None
        The ``--run-dir`` CLI value, if any.

    Returns
    -------
    str
        The run directory, by precedence ``--run-dir`` > ``$RUN_DIR`` >
        ``$OUTPUT_DIR/<partition>/<jobid>``.
    """
    if arg:
        return arg
    if os.environ.get("RUN_DIR"):
        return os.environ["RUN_DIR"]
    return os.path.join(os.environ.get("OUTPUT_DIR", "./results"),
                        os.environ.get("SLURM_JOB_PARTITION", "local"),
                        os.environ.get("SLURM_JOB_ID", "interactive"))


def main() -> None:
    """Parse CLI args, finalize the config, bootstrap distributed, and train.

    With ``--dry-run`` the finalized config and derived channel counts are
    printed and the function returns without importing beast or touching a GPU.
    Otherwise it initialises the process mesh and runs
    :func:`era5_levels.train.training_loop`.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="base YAML config")
    ap.add_argument("--overlay", action="append", default=[],
                    help="overlay YAML(s), applied in order (e.g. levels37.yaml)")
    ap.add_argument("--run-dir", help="stable dir for checkpoints+metrics; reused "
                    "across chained jobs to auto-resume (default $RUN_DIR or "
                    "$OUTPUT_DIR/<partition>/<jobid>)")
    ap.add_argument("--dry-run", action="store_true",
                    help="finalize+print the config and exit (no beast/GPU needed)")
    args = ap.parse_args()

    cfg = finalize_config(load_config(args.config, args.overlay))

    if args.dry_run:
        import yaml
        print(yaml.dump(cfg, sort_keys=False))
        print(f"=> n_variables={cfg['data']['n_variables']}  "
              f"in_channels={cfg['model']['n_input_channels']}  "
              f"out_channels={cfg['model']['n_output_channels']}")
        return

    cfg["run_dir"] = resolve_run_dir(args.run_dir)

    # distributed + process mesh
    _pg, rank, world, _local = beast_api.bootstrap_distributed(cfg["mesh_dims"])
    dist.barrier()
    if rank == 0:
        print(f"world={world}  mesh={cfg['mesh_dims']}  "
              f"levels={len(cfg['data']['pressure_levels'])}  run_dir={cfg['run_dir']}")

    from .train import training_loop
    training_loop(cfg)

    if rank == 0:
        print("done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
