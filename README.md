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

1. **Headline 13 vs 37** — train the matched pair, compare per-variable
   latitude-weighted RMSE at each level. (ACC is a natural add but needs a
   sharded climatology field — not implemented yet; RMSE is what's wired today.)
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
  beast_api.py         # the ONLY place that imports beast (model/Expert/comm/dataloader handles, dist bootstrap)
  checkpoint.py        # vendored sharded save/resume (the beast checkpoint fns moved to an unmerged PR)
  losses.py            # latitude-weighted MSE (deterministic loss)
  train.py             # deterministic training loop (adapted from BellBeast, beast APIs)
  evaluate.py          # DISTRIBUTED per-variable RMSE (shard-local + JSpatial/JChannel reductions) + subset indexing
  transfer.py          # frozen-core transfer (freeze all but patch_embedding/recovery)
  main.py              # entrypoint: bootstrap dist+mesh -> training_loop
configs/
  base_1p5.yaml        # shared core config for the headline 1.5° run (both levels)
  levels13_1p5.yaml    # overlay: pressure_levels (13) + 1.5° data paths
  levels37_1p5.yaml    # overlay: pressure_levels (37) + 1.5° data paths
  base_0p25.yaml       # shared core config for the full-resolution 0.25° run
  levels13.yaml / levels37.yaml   # 0.25° overlays
  smoke.yaml           # tiny 1-GPU config, dummy data, for a pipeline check
  smoke_nout2.yaml     # 1-GPU smoke for the 2-in/2-out (multi-step) path
slurm/
  setup_env.sh         # venv on a workspace + editable beast (+ its jigsaw submodule) + this repo
  submit_smoke.sh      # 1-GPU end-to-end smoke job
  submit_train_1p5.sh  # multi-GPU 1.5° training (HoreKa H200), stable RUN_DIR + auto-resume
  submit_train.sh      # multi-GPU 0.25° training, stable RUN_DIR + auto-resume
  submit_chain.sh      # chain of afterany jobs sharing RUN_DIR — survives time-limit kills
  coarsen_*.sh / submit_subset.sh   # data-prep jobs (coarsen to 1.5°, carve the 13-level subset)
scripts/
  coarsen_to_1p5.py    # block-mean a 0.25° zarr to 1.5° (resumable)
  coarsen_masks.py     # block-mean the constant masks the same way
  make_level_subset.py # carve the 13-level store from the 37-level one (+ subset norm)
  run_subset_eval.py   # free 37→13 comparison — distributed entrypoint (+ --check-indices)
  plot_results.py      # offline poster figures from metrics.csv + map dumps (laptop, no beast)
```

## Setup & run (HoreKa)

```bash
# 0. install (once): venv on a workspace, editable beast (+ jigsaw submodule) + this repo
export WS=$(ws_find levels)
export BEAST_DIR=$HOME/beast          # your beast checkout
bash slurm/setup_env.sh
# setup_env.sh does the full sequence; the beast part is, in effect:
#   git -C $BEAST_DIR submodule update --init --recursive   # jigsaw is a submodule
#   pip install -e $BEAST_DIR                                # beast (pulls torch_blue)
#   pip install -e $BEAST_DIR/libs/jigsaw                    # jigsaw (top-level import,
#                                                            #   NOT a beast pyproject dep)

# 1. prove the pipeline runs end-to-end (random dummy data, tiny model)
sbatch slurm/submit_smoke.sh

# 2. train the matched pair (edit data paths in the overlays + partition/nodes first).
#    The headline runs at 1.5° (base_1p5 + levels{13,37}_1p5) — affordable to
#    train to compute-parity by the deadline; swap in base_0p25 + levels{13,37}
#    for the full-resolution run. Use submit_chain.sh for long runs: each link
#    auto-resumes from the shared RUN_DIR, so SLURM time-limit kills don't cost
#    progress. The two chains run in parallel (independent jobs).
export DATA_DIR=/path/to/zarr_root ; export WORKDIR=$(pwd)
bash slurm/submit_chain.sh configs/base_1p5.yaml configs/levels13_1p5.yaml 4
bash slurm/submit_chain.sh configs/base_1p5.yaml configs/levels37_1p5.yaml 4
# (or a single job each: sbatch slurm/submit_train_1p5.sh configs/base_1p5.yaml configs/levels13_1p5.yaml)

# 3. headline comparison (DISTRIBUTED, same mesh as training, after both finish)
python scripts/run_subset_eval.py --check-indices    # pure-python channel-index sanity
srun python -u scripts/run_subset_eval.py \
    --config configs/base_1p5.yaml --overlay configs/levels37_1p5.yaml \
    --results-dir $WS/results/levels37

# 4. poster figures (offline, on your laptop — no beast/GPU)
pip install -e ".[plots]"
python scripts/plot_results.py \
    --csv13 $WS/results/levels13/metrics.csv \
    --csv37 $WS/results/levels37/metrics.csv \
    --maps-dir $WS/results/levels37/maps/epoch_0 --out figures/
```

The only config difference between the two runs is the overlay
(`pressure_levels` + data paths). Inspect the *finalized* config any time without
a GPU/beast:

```bash
python -m era5_levels.main --config configs/base_1p5.yaml \
    --overlay configs/levels37_1p5.yaml --dry-run
```

## Outputs (for the poster)

Each validation pass (rank 0) writes to the run's results dir:

- **`metrics.csv`** — long-format `epoch, variable, model, persistence,
  climatology` (latitude-weighted RMSE per variable-at-level). Load with pandas
  and pivot for the per-level curves and the 13-vs-37 improvement heatmap; the
  **persistence** and (self-computed, in-sample) **climatology** columns give the
  baselines that make those plots meaningful.
- **`maps/epoch_<n>/<var>_{pred,true,err}.npy`** — full `(lat, lon)` fields for
  the variables listed in `training.dump_maps_vars`, for the forecast/error map
  figures. Gathered over `JSpatial`; nothing is gathered across channels.

## Configuration notes

- **Channels are derived, never set by hand.** `config.py` computes
  `n_variables`, `n_input_channels`, `n_output_channels` from `pressure_levels`
  via `variable_layout.num_variables`. That's what makes 13↔37 a one-field change.
- **`mesh_dims`** = `[experts, dp, up, jigsaw-spatial, jigsaw-channel]`; the
  product must equal your GPU count. We use **experts=1** (universal expert →
  avoids the per-expert hardcoded level slicing) and **up=1** (deterministic, no
  sample parallelism). At 0.25° we use `[1,1,1,4,2]` = 8 GPUs; at 1.5° the
  240-lon grid won't tile with `jspatial=4` (240/16=15 has no even divisor), so
  the 1.5° config scales via data parallelism instead — `[1,2,1,2,2]` = 8 GPUs.
- **Speed levers at 0.25°:** the default window sizes are the hero-proven
  divisible values for this grid (guaranteed to tile) but are large
  (near-global). Once a run works, **shrink the windows** (biggest lever) and/or
  lower `embedding_dimension`; re-check divisibility on the distributed layout.
  On H200/Ruby you have memory headroom to push `embedding_dimension` up instead.
- **37-level order:** confirm `PRESSURE_LEVELS_37` (in `variable_layout.py` and
  `levels37.yaml`) matches your 37-level zarr's `feature` coordinate and your
  `norm_mean.npy`/`norm_std.npy` ordering before training.

---

## Status & where it runs

The model and data are sharded across GPUs (channels over `JChannel`, longitude
over `JSpatial`), so the full field never exists on a single rank and the
pipeline **cannot be built, trained, or evaluated off-cluster** — not even at
reduced scale. That covers everything that matters: model build, dataloader, the
training step, and **all evaluation** (metrics are computed on local shards and
reduced — spatial means `all_reduce`'d over `JSpatial`, per-variable RMSE
`all_gather`'d over `JChannel`, mirroring beast's own reductions).

The *only* things that execute off-cluster are a couple of **pure-Python sanity
checks** — `--dry-run` (config + channel arithmetic) and `--check-indices` (the
37→13 channel index map). They confirm bookkeeping, nothing more.

On the cluster the pipeline **has run end-to-end**: `submit_smoke.sh` is green,
and the deterministic loop reaches steady training on real ERA5 (model build →
dataloader → training step → checkpoint/resume all exercised). Treat
`submit_smoke.sh` as the integration test and get it green after any beast bump;
**`beast_api.py` is the single place** to fix an import/signature if beast moves
something.

This repo intentionally does **not** vendor or modify beast's model/dataloader —
those come from your `pip install -e` checkout, so beast fixes flow in for free.
The one exception is `checkpoint.py` (vendored, see its module docstring) because
the sharded save/resume helpers were dropped from `beast.utils` in the refactor.
For evaluation we use the small local `evaluate.py`; adopt `beast.evaluation`
once that subpackage stabilises.

### Resolution: 0.25° vs 1.5°

The *fair-comparison* argument is resolution-independent — pressure levels live
on the channel axis, so coarsening longitude/latitude is orthogonal to the
13-vs-37 question. The headline trained comparison therefore runs at **1.5°**
(`configs/base_1p5.yaml` + `levels{13,37}_1p5.yaml`, launched with
`slurm/submit_train_1p5.sh`), which is affordable to train to compute-parity by
the deadline while keeping all 37 vertical levels. The 0.25° configs
(`base_0p25.yaml` + `levels{13,37}.yaml`) are kept for the full-resolution run if
compute allows. Coarsen your 0.25° 37-level zarr to 1.5° with
`scripts/coarsen_to_1p5.py` (and the masks with `scripts/coarsen_masks.py`); the
13-level store is carved from the 37-level one with `scripts/make_level_subset.py`.

### Multi-step (`n_in=2` / `n_out=2`)

`config.py` derives `n_input_channels` / `n_output_channels` from
`n_in_timesteps` / `n_out_timesteps`, so 2-in/2-out works as a config change
(`configs/smoke_nout2.yaml` is the single-GPU smoke for it). Note: the local
`evaluate.py` scores the variables as laid out on the channel axis; for
`n_out_timesteps>1` the output channels are `[timestep-0 vars, timestep-1 vars]`,
so per-variable RMSE is reported per output timestep — interpret the metric rows
accordingly.
