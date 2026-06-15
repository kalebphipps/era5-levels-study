"""Free 13-vs-37 subset evaluation — DISTRIBUTED, run on the cluster.

Because the model + data are sharded across GPUs (jigsaw: channels over JChannel,
longitude over JSpatial), this cannot run on a single GPU. It launches inside the
same process mesh as training, rebuilds the (sharded) model, loads the sharded
checkpoint for each rank, and computes per-variable RMSE reduced across the mesh
— restricted to the 13 standard levels so a trained 37-level model is scored on
exactly the variables/levels the 13-level model predicts. No retraining.

Run it like a training job (one task per GPU, same mesh_dims as the run you are
evaluating):

    srun python -u scripts/run_subset_eval.py \\
        --config configs/base_0p25.yaml --overlay configs/levels37.yaml \\
        --results-dir $WS/results/<partition>/<jobid>

Index-only sanity check (pure Python, no GPU/beast — just the channel maths):

    python scripts/run_subset_eval.py --check-indices
"""

from __future__ import annotations

import argparse

import torch

from era5_levels.variable_layout import PRESSURE_LEVELS_13, PRESSURE_LEVELS_37


def check_indices():
    """Print where the 13 standard levels land inside the 37-level layout."""
    from era5_levels.evaluate import subset_indices
    idx = subset_indices(PRESSURE_LEVELS_37, PRESSURE_LEVELS_13)
    print(f"37-level layout has {6 + 6 * len(PRESSURE_LEVELS_37)} channels; "
          f"13-level subset selects {len(idx)} of them.")
    print("first/last subset channel indices:", idx[:6].tolist(), "...", idx[-3:].tolist())


def run_distributed(args):
    """Rebuild the sharded model, load its checkpoint, and score the 13-level subset.

    Runs inside the same process mesh as training (one task per GPU). Per-variable
    RMSE is reduced across the mesh and restricted to the 13 standard levels, then
    written to ``<results-dir>/subset_metrics.csv`` on rank 0.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments (``config``, ``overlay``, ``results_dir``, ...).
    """
    import torch.distributed as dist

    from era5_levels import beast_api
    from era5_levels.config import finalize_config, load_config
    from era5_levels.evaluate import validate_per_level
    from era5_levels.train import build_model

    cfg = finalize_config(load_config(args.config, args.overlay))
    beast_api.bootstrap_distributed(cfg["mesh_dims"])
    _, get_pg = beast_api.get_comm()
    groups = {n: get_pg(n) for n in
              ("JSpatial", "JChannel", "DTP", "SP", "DP", "DDP", "Expert")}
    utils = beast_api.get_utils()
    Expert = beast_api.get_expert_class()
    get_dataloader = beast_api.get_dataloader_fn()

    dtype = utils.set_dtype(cfg["training"]["dtype"])
    device = torch.device("cuda", torch.cuda.current_device())

    model = build_model(cfg, device, dtype, groups)
    model = Expert(model, groups["Expert"])

    # load this rank's checkpoint shard (beast saves per ep/jchannel rank)
    state = utils.load_latest_model_states(args.results_dir)[0]
    # strip a possible DDP "module." prefix
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=False)

    valid_dl = get_dataloader(cfg, groups["DTP"].rank(), groups["DTP"].size(),
                              groups["DP"].rank(), groups["DP"].size(),
                              mode="validation", dtype=dtype)

    # per-variable std (physical units) if available on the dataset
    ds = getattr(valid_dl, "dataset", None)
    data_std = getattr(ds, "norm_values", {}).get("std") if ds is not None else None

    names, rmse = validate_per_level(
        model, valid_dl, cfg["data"]["pressure_levels"], groups, device,
        subset_levels=PRESSURE_LEVELS_13, data_std=data_std,
    )
    if dist.get_rank() == 0:
        print(f"{'variable':32s} {'rmse':>12s}")
        for n, v in zip(names, rmse.tolist()):
            print(f"{n:32s} {v:12.4f}")
        print(f"\nmean RMSE over 13 common levels: {rmse.mean().item():.4f}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-indices", action="store_true",
                    help="pure-Python channel-index sanity check (no GPU/beast)")
    ap.add_argument("--config")
    ap.add_argument("--overlay", action="append", default=[])
    ap.add_argument("--results-dir", help="run dir containing checkpoints/ to evaluate")
    args = ap.parse_args()

    if args.check_indices:
        check_indices()
    elif args.config and args.results_dir:
        run_distributed(args)
    else:
        ap.error("give --check-indices, or --config + --results-dir for the "
                 "distributed evaluation.")
