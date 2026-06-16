"""Launch experiments."""

from __future__ import annotations

import argparse
import os

import torch.distributed as dist

from . import beast_api
from .config import finalize_config, load_config


def resolve_run_dir(arg: str | None) -> str:
    """Resolve output dir for checkpoints and metrics.

    Parameters
    ----------
    arg : str or None
        The ``--run-dir`` value, if any.

    Returns
    -------
    str
        The run directory.
    """
    if arg:
        return arg
    if os.environ.get("RUN_DIR"):
        return os.environ["RUN_DIR"]
    return os.path.join(os.environ.get("OUTPUT_DIR", "./results"),
                        os.environ.get("SLURM_JOB_PARTITION", "local"),
                        os.environ.get("SLURM_JOB_ID", "interactive"))


def main() -> None:
    """Parse args, finalize the config, set up distributed, and train."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="base YAML config")
    ap.add_argument("--overlay", action="append", default=[],
                    help="overlay YAML(s), applied in order (e.g. levels37.yaml)")
    ap.add_argument("--run-dir", help="dir for checkpoints+metrics")
    ap.add_argument("--dry-run", action="store_true",
                    help="finalize and print the config and exit.")
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

    # distributed and process mesh
    _pg, rank, world, _local = beast_api.bootstrap_distributed(cfg["mesh_dims"])
    dist.barrier()
    if rank == 0:
        print(f"world={world}  mesh={cfg['mesh_dims']}  "
              f"levels={len(cfg['data']['pressure_levels'])}  run_dir={cfg['run_dir']}")

    # Import here to allow quick dry run checks.
    from .train import training_loop
    training_loop(cfg)

    if rank == 0:
        print("done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
