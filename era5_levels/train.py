"""Deterministic training loop for the 13-vs-37 level study.

Adapted from the reference (BellBeast) ``training.py`` but:
  * deterministic only — no Bayesian/VI, no NICE noise, no ensemble samples;
  * targets beast's current APIs via `beast_api` (get_dataloader positional
    ranks, comm mesh, 2-arg Expert) instead of BellBeast's signatures;
  * loss = latitude-weighted MSE (`LatitudeWeightedMSE`);
  * in-loop validation = scalar weighted-MSE (distributed-safe); per-level RMSE
    and the free subset-eval are done offline by ``scripts/run_subset_eval.py``;
  * optional frozen-core transfer (freeze everything but the I/O convs).

This is the boilerplate scaffold: the structure and API calls follow the code
that exists in beast/BellBeast, but it can only actually run on the cluster with
beast (+ jigsaw) installed and a process group launched. Treat the first cluster
run as a debugging pass.
"""

from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from . import beast_api
from .evaluate import dump_sample_maps, evaluate_all, write_metrics_csv
from .losses import LatitudeWeightedMSE
from .transfer import freeze_core, load_core_from_checkpoint


def build_model(cfg, device, dtype, groups):
    Bellbeast, TinyBellbeast = beast_api.get_model_classes()
    m, mdl = cfg["model"], cfg["data"]
    common = dict(
        xlat=mdl["xlat"], xlon=mdl["xlon"],
        in_channels=m["n_input_channels"], embed_dim=m["embedding_dimension"],
        out_channels=m["n_output_channels"], heads=m["n_attn_heads"],
        kernel_size=m["patch_size"], window_size_outer=m["window_size_outer"],
        window_size_inner=m["window_size_inner"], n_full_blocks=m["n_attn_blocks"],
        noise_dim=m["noise_dim"], device=device, bayesian=False, dtype=dtype,
    )
    if mdl["xlat"] == 120 and mdl["xlon"] == 240:
        model = TinyBellbeast(
            channel_group=groups["JChannel"], ep_group=groups["Expert"],
            up_group=groups["SP"], **common,
        )
    else:
        model = Bellbeast(
            dtp_rank=groups["DTP"].rank(), dtp_group=groups["DTP"],
            spatial_group=groups["JSpatial"], channel_group=groups["JChannel"],
            ep_group=groups["Expert"], up_group=groups["SP"],
            parallelism=cfg["jigsaw"]["parallelism"],
            flash_attn=cfg["training"]["flash_attn"], nice_norm=m["nice_norm"],
            **common,
        )
    return model.to(dtype).to(device)


def training_loop(cfg):
    _, get_pg = beast_api.get_comm()
    groups = {n: get_pg(n) for n in
              ("JSpatial", "JChannel", "DTP", "SP", "DP", "DDP", "Expert")}
    utils = beast_api.get_utils()
    Expert = beast_api.get_expert_class()
    get_dataloader = beast_api.get_dataloader_fn()

    dtype = utils.set_dtype(cfg["training"]["dtype"])
    device = torch.device("cuda", torch.cuda.current_device())
    is_root = dist.get_rank() == 0

    # deterministic, per-rank-varied init seed (mirrors reference)
    utils.set_all_seeds(3 + groups["JChannel"].rank()
                        + groups["Expert"].rank() * groups["JChannel"].size())

    if is_root:
        print("Building model...")
    model = build_model(cfg, device, dtype, groups)

    # universal expert (ep_group size 1 -> all-ones loss weights for ANY channel
    # count, so 37 levels works without the per-expert hardcoded slicing)
    model = Expert(model, groups["Expert"])
    expert_weights = model.get_loss_weights()

    # optional frozen-core transfer: load a trained core, freeze all but I/O
    transfer_cfg = cfg.get("transfer") or {}
    if transfer_cfg.get("source_checkpoint"):
        ckpt = torch.load(transfer_cfg["source_checkpoint"], map_location="cpu",
                          weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        loaded = load_core_from_checkpoint(model, sd)
        n_tr, n_fr = freeze_core(model)
        if is_root:
            print(f"[transfer] loaded {len(loaded)} core tensors; "
                  f"trainable={n_tr:,} frozen={n_fr:,}")

    model = DistributedDataParallel(
        model, device_ids=[device.index], output_device=device.index,
        process_group=groups["DDP"], gradient_as_bucket_view=True,
        static_graph=True, bucket_cap_mb=50,
    )

    # two LR groups: low LR for the conv encoder/decoder, normal for the core
    low_keys = ("patch_embedding", "downsample", "upsample", "patch_recovery")
    low, normal = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (low if any(k in name for k in low_keys) else normal).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": low, "lr": cfg["training"]["low_lr"]},
         {"params": normal, "lr": cfg["training"]["normal_lr"]}],
        betas=(0.9, 0.9), eps=1e-6, weight_decay=0,
    )
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["n_epochs"], eta_min=1e-5)
        if cfg["training"]["lr_scheduler"] else None)

    if is_root:
        print("Building dataloaders...")
    train_dl = get_dataloader(cfg, groups["DTP"].rank(), groups["DTP"].size(),
                              groups["DP"].rank(), groups["DP"].size(),
                              mode="train", dtype=dtype)
    valid_dl = None if cfg.get("skip_validation") else get_dataloader(
        cfg, groups["DTP"].rank(), groups["DTP"].size(),
        groups["DP"].rank(), groups["DP"].size(), mode="validation", dtype=dtype)

    loss_fn = LatitudeWeightedMSE()
    sqrt_w = torch.sqrt(utils.get_spatial_weights(cfg, device))

    results_path = cfg.get("run_dir") or os.path.join(
        os.environ.get("OUTPUT_DIR", "./results"),
        os.environ.get("SLURM_JOB_PARTITION", "local"),
        os.environ.get("SLURM_JOB_ID", "interactive"))
    ckpt_path = os.path.join(results_path, "checkpoints")
    if is_root:
        os.makedirs(ckpt_path, exist_ok=True)
    dist.barrier()

    # Auto-resume: if this run dir already holds checkpoints (e.g. a chained job
    # restarting after a SLURM time-limit kill), reload the latest shard for this
    # rank and continue. Fresh runs (empty dir) just start at epoch 0.
    start_epoch = 0
    has_ckpt = os.path.isdir(ckpt_path) and any(
        f.endswith(".pt") for f in os.listdir(ckpt_path))
    if has_ckpt:
        try:
            msd, osd, last_epoch, *_ = utils.load_latest_model_states(results_path)
            model, optimizer, start_epoch = utils.load_state_dict(
                model, optimizer, msd, osd, last_epoch)
            if is_root:
                print(f"[resume] loaded epoch {last_epoch} -> starting at {start_epoch}")
        except Exception as e:  # noqa: BLE001 - resume is best-effort
            if is_root:
                print(f"[resume] failed ({e}); starting fresh")

    for epoch in range(start_epoch, cfg["training"]["n_epochs"]):
        if scheduler is not None:
            scheduler.step(epoch)
        t0 = time.perf_counter()
        avg = train_one_epoch(cfg, epoch, train_dl, model, optimizer, loss_fn,
                              sqrt_w, device, groups["DTP"], ckpt_path, utils)
        if is_root:
            print(f"epoch {epoch}: train weighted-MSE {avg:.5f} "
                  f"({(time.perf_counter() - t0) / 60:.1f} min)")
        utils.save_checkpoint(ckpt_path, model, optimizer, epoch, 0, avg)

        if valid_dl is not None:
            vloss = validate(valid_dl, model, loss_fn, sqrt_w, device, groups["DTP"])
            if is_root:
                print(f"epoch {epoch}: valid weighted-MSE {vloss:.5f}")
            # distributed per-variable RMSE + persistence/climatology baselines
            # (reduced over JSpatial, gathered over JChannel — the full field is
            # never gathered onto one rank)
            names, metrics = evaluate_all(
                model, valid_dl, cfg["data"]["pressure_levels"], groups, device,
                n_in_timesteps=cfg["data"]["n_in_timesteps"], baselines=True)
            if is_root:
                msg = "  ".join(f"{k}={v.mean().item():.4f}" for k, v in metrics.items())
                print(f"epoch {epoch}: mean RMSE  {msg}")
                # long-format CSV for offline plots (per-level curves, heatmaps)
                write_metrics_csv(os.path.join(results_path, "metrics.csv"),
                                  epoch, names, metrics)

            # optional sample-field maps for the poster (gathered over JSpatial)
            dump_vars = cfg["training"].get("dump_maps_vars")
            if dump_vars:
                dump_sample_maps(
                    model, valid_dl, cfg["data"]["pressure_levels"], groups,
                    device, dump_vars,
                    os.path.join(results_path, "maps", f"epoch_{epoch}"))


def train_one_epoch(cfg, epoch, dl, model, optimizer, loss_fn, sqrt_w, device,
                    dtp_group, ckpt_path, utils):
    model.train()
    autocast = cfg["training"]["autocast"]
    ckpt_every = cfg["training"]["checkpoint_interval"]
    loss_sum = torch.zeros(1, device=device)
    if hasattr(dl, "sampler") and hasattr(dl.sampler, "set_epoch"):
        dl.sampler.set_epoch(epoch)

    for i, batch in enumerate(dl):
        x, y = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            pred = model(x)
            loss = loss_fn(pred, y, sqrt_w)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        loss_sum += loss.detach()
        if ckpt_every and i % ckpt_every == 0:
            utils.save_checkpoint(ckpt_path, model, optimizer, epoch, i, loss_sum)
        if i % 50 == 0 and dist.get_rank() == 0:
            print(f"  epoch {epoch} step {i}/{len(dl)} loss {float(loss):.5f}")

    dist.all_reduce(loss_sum, group=dtp_group)
    return loss_sum.item() / max(1, len(dl)) / dist.get_world_size(dtp_group)


@torch.no_grad()
def validate(dl, model, loss_fn, sqrt_w, device, dtp_group):
    model.eval()
    loss_sum = torch.zeros(1, device=device)
    for batch in dl:
        x, y = batch[0].to(device), batch[1].to(device)
        pred = model(x)
        loss_sum += loss_fn(pred, y, sqrt_w).detach()
    dist.all_reduce(loss_sum, group=dtp_group)
    return loss_sum.item() / max(1, len(dl)) / dist.get_world_size(dtp_group)
