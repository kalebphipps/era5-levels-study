# ERA5 pressure-level study: does 13 → 37 levels help?

A small, poster-scoped study: train a **deterministic** SWIN weather model on
ERA5 at **0.25°** with **13** vs **37** pressure levels and ask whether the extra
vertical resolution improves forecast skill — under a *matched, fair* comparison.

Built **on top of `beast`** (the model + dataloader come from there, installed
separately). This repo holds only the study-specific glue: config/channel
derivation, a deterministic training loop, evaluation, and the frozen-core
transfer ablation. The Bayesian/VI machinery is deliberately ignored.

---

## The idea & why the comparison is fair

Pressure levels live on the **channel axis**. In the model, only two layers
depend on the channel count:

- `patch_embedding` (in_channels → embed) and
- `patch_recovery` (embed → out_channels).

**Everything else — the entire SWIN core (downsample, transformer blocks,
upsample, mix) — runs in `embed_dim` space and is architecturally identical**
between the 13- and 37-level models. So we keep `embedding_dimension`,
`n_attn_blocks`, `n_attn_heads`, `patch_size`, and the window sizes **byte-
identical** across the two runs; only `data.pressure_levels` changes (84 vs 228
channels, derived automatically). The cores have identical parameter counts and
near-identical per-step compute, so the difference you measure is *the levels*,
not the model. Train both to the **same compute budget** (same epochs/steps) and
compare — you don't need convergence, you need parity.

## Three results, increasing cost

1. **Headline 13 vs 37** — train the matched pair, compare per-level RMSE/ACC.
2. **Free subset-eval** — score the 37-level model on *exactly* the 13 standard
   levels (a subset of its outputs) vs the 13-level model. No extra training.
   Runs **distributed in the mesh** (the field is sharded and can't be gathered
   to one GPU); see [`scripts/run_subset_eval.py`](scripts/run_subset_eval.py).
3. **Frozen-core transfer** — freeze a trained core, retrain *only* the two I/O
   convs for the other level count. Isolates whether benefit lives in the
   representation or the I/O. Cheap (≈2 conv layers). Set `transfer.source_checkpoint`
   in the config; see [`era5_levels/transfer.py`](era5_levels/transfer.py).

---

## Layout

```
era5_levels/
  variable_layout.py   # vendored: the 13/37 channel-order source of truth (stdlib only)
  config.py            # load YAML + derive channel counts from pressure_levels
  beast_api.py         # the ONLY place that imports beast (beast→gb fallback, dist bootstrap)
  losses.py            # latitude-weighted MSE (deterministic loss)
  train.py             # deterministic training loop (adapted from BellBeast, beast APIs)
  evaluate.py          # study-local per-level RMSE/ACC + subset-index helper
  transfer.py          # frozen-core transfer (freeze all but patch_embedding/recovery)
  main.py              # entrypoint: bootstrap dist+mesh -> training_loop
configs/
  base_0p25.yaml       # shared core config (identical for both runs)
  levels13.yaml        # overlay: pressure_levels (13) + data paths
  levels37.yaml        # overlay: pressure_levels (37) + data paths
  smoke.yaml           # tiny 1-GPU config, dummy data, for a pipeline check
slurm/
  setup_env.sh         # venv on a workspace + pip install -e beast + this repo
  submit_smoke.sh      # 1-GPU end-to-end smoke job
  submit_train.sh      # multi-GPU training (HoreKa TEAL/Ruby)
scripts/
  run_subset_eval.py   # the free 37→13 comparison
```

## Setup & run (HoreKa)

```bash
# 0. install (once): venv on a workspace, editable beast + this repo
export WS=$(ws_find levels)
export BEAST_DIR=$HOME/beast          # your beast checkout (you pip install -e it)
bash slurm/setup_env.sh

# 1. prove the pipeline runs end-to-end (random dummy data, tiny model)
sbatch slurm/submit_smoke.sh

# 2. train the matched pair (edit data paths in the overlays + partition/nodes first)
export DATA_DIR=/path/to/zarr_root ; export WORKDIR=$(pwd)
sbatch slurm/submit_train.sh configs/base_0p25.yaml configs/levels13.yaml
sbatch slurm/submit_train.sh configs/base_0p25.yaml configs/levels37.yaml

# 3. headline comparison (DISTRIBUTED, same mesh as training, after both finish)
python scripts/run_subset_eval.py --check-indices    # pure-python channel-index sanity
srun python -u scripts/run_subset_eval.py \
    --config configs/base_0p25.yaml --overlay configs/levels37.yaml \
    --results-dir $WS/results/<partition>/<jobid>
```

The only config difference between the two runs is the overlay
(`pressure_levels` + data paths). Inspect the *finalized* config any time without
a GPU/beast:

```bash
python -m era5_levels.main --config configs/base_0p25.yaml \
    --overlay configs/levels37.yaml --dry-run
```

## Configuration notes

- **Channels are derived, never set by hand.** `config.py` computes
  `n_variables`, `n_input_channels`, `n_output_channels` from `pressure_levels`
  via `variable_layout.num_variables`. That's what makes 13↔37 a one-field change.
- **`mesh_dims`** = `[experts, dp, up, jigsaw-spatial, jigsaw-channel]`; the
  product must equal your GPU count. We use **experts=1** (universal expert →
  avoids the per-expert hardcoded level slicing) and **up=1** (deterministic, no
  sample parallelism). Default `[1,1,1,4,2]` = 8 GPUs.
- **Speed levers at 0.25°:** the default window sizes are the hero-proven
  divisible values for this grid (guaranteed to tile) but are large
  (near-global). Once a run works, **shrink the windows** (biggest lever) and/or
  lower `embedding_dimension`; re-check divisibility on the distributed layout.
  On H200/Ruby you have memory headroom to push `embedding_dimension` up instead.
- **37-level order:** confirm `PRESSURE_LEVELS_37` (in `variable_layout.py` and
  `levels37.yaml`) matches your 37-level zarr's `feature` coordinate and your
  `norm_mean.npy`/`norm_std.npy` ordering before training.

---

## What's validated vs. what needs the cluster (be honest with yourself)

**Validated locally** (pure Python, no beast/GPU): config loading + channel
derivation (`--dry-run`), the level layout, the frozen-core freeze/load logic,
and the subset channel-index maths (`run_subset_eval.py --check-indices`).

**Inherently distributed — only runs on the cluster, in the mesh** (cannot run
off-cluster, and the field cannot be gathered onto one GPU): the model build,
dataloader, training step, **and all evaluation**. Metrics are computed on local
shards and reduced across the groups (spatial means `all_reduce`'d over
`JSpatial`, per-variable results `all_gather`'d over `JChannel`) — mirroring
beast's own reductions; swap to `beast.evaluation` once it stabilises. The code
targets the beast/BellBeast APIs as they exist today, but beast is mid-rename
(`gb` → `beast`) and mid-refactor; **`beast_api.py` is the single place** to
adjust if an import or signature has moved. Treat `submit_smoke.sh` as the
integration test: get it green before launching the 0.25° pair.

This repo intentionally does **not** vendor or modify beast's model/dataloader —
those come from your `pip install -e` checkout, so beast fixes flow in for free.
The one beast-side change the study benefits from (generalizing the deprecated
top-level `evaluation.py`'s hardcoded 13-level list) is sidestepped here by using
the small local `evaluate.py`; adopt `beast.evaluation` once it stabilizes.
