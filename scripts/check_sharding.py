"""Check sharding is correct."""

from __future__ import annotations

import argparse


def main() -> None:
    """Build the model and dataloader in the mesh and check the layout assumptions."""
    import torch
    import torch.distributed as dist

    from era5_levels import beast_api
    from era5_levels.config import finalize_config, load_config
    from era5_levels.evaluate import evaluate_all, persistence_prediction
    from era5_levels.train import build_model
    from era5_levels.variable_layout import num_variables

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overlay", action="append", default=[])
    ap.add_argument("--n-batches", type=int, default=50,
                    help="validation samples used for the baseline comparison")
    args = ap.parse_args()

    cfg = finalize_config(load_config(args.config, args.overlay))
    cfg["data"]["valid_subset"] = args.n_batches

    beast_api.bootstrap_distributed(cfg["mesh_dims"])
    _, get_pg = beast_api.get_comm()
    groups = {n: get_pg(n) for n in
              ("JSpatial", "JChannel", "DTP", "SP", "DP", "DDP", "Expert")}
    utils = beast_api.get_utils()
    Expert = beast_api.get_expert_class()
    get_dataloader = beast_api.get_dataloader_fn()

    dtype = utils.set_dtype(cfg["training"]["dtype"])
    device = torch.device("cuda", torch.cuda.current_device())
    is_root = dist.get_rank() == 0
    n_in = cfg["data"]["n_in_timesteps"]
    n_out = cfg["data"]["n_out_timesteps"]
    levels = cfg["data"]["pressure_levels"]

    if is_root:
        print(f"mesh={cfg['mesh_dims']} levels={len(levels)} -- building model + "
              "dataloader...", flush=True)
    model = Expert(build_model(cfg, device, dtype, groups), groups["Expert"])
    dl = get_dataloader(cfg, groups["DTP"].rank(), groups["DTP"].size(),
                        groups["DP"].rank(), groups["DP"].size(),
                        mode="validation", dtype=dtype)

    # GBDataLoader yields (input, target, start_ns, end_ns) -- index like the
    # training/eval loops do, don't unpack.
    batch = next(iter(dl))
    x, y = batch[0].to(device), batch[1].to(device)
    local_out = y.shape[1]
    fails: list[str] = []

    sp = groups["JSpatial"]
    sp_size = dist.get_world_size(sp)
    if sp_size > 1:
        lon = torch.tensor([x.shape[-1]], device=device)
        sizes = [torch.zeros_like(lon) for _ in range(sp_size)]
        dist.all_gather(sizes, lon, group=sp)
        sizes = [int(s.item()) for s in sizes]
        ok = len(set(sizes)) == 1
        if is_root:
            print(f"[1] JSpatial lon shards {sizes} -> {'EQUAL ok' if ok else 'UNEQUAL!!'}")
        if not ok:
            fails.append("unequal JSpatial shards -> lat-weighted mean is biased")
    elif is_root:
        print("[1] JSpatial size 1 (longitude not split) -> n/a")

    jc = dist.get_world_size(groups["JChannel"])
    expected_out = n_out * num_variables(levels)
    ok2 = local_out * jc == expected_out
    if is_root:
        print(f"[2] output channels: local {local_out} x jchannel {jc} = "
              f"{local_out * jc} (expect {expected_out}) -> {'ok' if ok2 else 'MISMATCH!!'}")
    if not ok2:
        fails.append("channel split does not tile the output variables")

    mask_ch = x.shape[1] - n_in * local_out
    ok3 = mask_ch >= 0
    if is_root:
        print(f"[3] input channels {x.shape[1]} = n_in({n_in}) x local_vars"
              f"({local_out}) + masks({mask_ch}) -> {'ok' if ok3 else 'BAD!!'}")
    if not ok3:
        fails.append("input channels < n_in * local_vars -> layout assumption wrong")
    persist = persistence_prediction(x, n_in, local_out)
    ok3b = tuple(persist.shape) == tuple(y.shape)
    if is_root:
        print(f"[3b] persistence slice {tuple(persist.shape)} vs target "
              f"{tuple(y.shape)} -> {'ok' if ok3b else 'BAD!!'}")
    if not ok3b:
        fails.append("persistence slice shape != target shape")

    # [3c] DEFINITIVE layout check: the persistence slice (last input timestep)
    # and the target are the SAME variables 6h apart, so each channel's spatial
    # field is strongly correlated with the target's. A misaligned slice (wrong
    # variables) gives ~0. This is robust to the tiny-window climatology artefact
    # that makes persistence-vs-climatology unreliable on a short subset.
    pf = persist[0].float().reshape(local_out, -1)
    yf = y[0].float().reshape(local_out, -1)
    pf = pf - pf.mean(dim=1, keepdim=True)
    yf = yf - yf.mean(dim=1, keepdim=True)
    corr = (pf * yf).sum(1) / (pf.norm(dim=1) * yf.norm(dim=1) + 1e-8)
    local_mean_corr = torch.nan_to_num(corr, nan=0.0).mean()
    dist.all_reduce(local_mean_corr, op=dist.ReduceOp.SUM)
    mean_corr = (local_mean_corr / dist.get_world_size()).item()
    ok3c = mean_corr > 0.3
    if is_root:
        print(f"[3c] persistence<->target per-channel spatial corr = {mean_corr:.3f} "
              f"-> {'ok (aligned)' if ok3c else 'FAIL (layout misaligned; ~0 expected if wrong)'}")
    if not ok3c:
        fails.append("persistence/target correlation ~0 -> input channel layout misaligned")

    if is_root:
        print(f"[4] running evaluate_all over {args.n_batches} samples...", flush=True)
    names, res = evaluate_all(model, dl, levels, groups, device,
                              n_in_timesteps=n_in, baselines=True)
    if is_root:
        gathered_ok = all(r.numel() == len(names) for r in res.values())
        finite = all(bool(torch.isfinite(r).all()) for r in res.values())
        mm = res["model"].mean().item()
        pm = res["persistence"].mean().item()
        cm = res["climatology"].mean().item()
        print(f"[4] gathered len == #names: {gathered_ok}; all finite: {finite}")
        print(f"    mean RMSE  model={mm:.3f}  persistence={pm:.3f}  climatology={cm:.3f}")
        print("    (note: over this tiny subset climatology is a ~2-day mean and"
              " unusually strong; persistence<climatology only holds over the full"
              " year, so it is NOT used as a pass/fail signal here -- [3c] is.)")
        if not gathered_ok:
            fails.append("gathered RMSE length != number of variables")
        if not finite:
            fails.append("non-finite RMSE")

        print("=" * 64)
        if fails:
            print("SHARDING CHECK FAILED:")
            for f in fails:
                print("   - " + f)
        else:
            print("ALL SHARDING CHECKS PASSED")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
