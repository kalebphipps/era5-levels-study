# ERA5 Pressure Levels Comparison

This repository contains code to compare training a deterministic SWIN weather prediction model on ERA5 with both the
standard 13 and expanded 37 pressure levels. The model configuration is kept as fair as possible with the focus on
whether  vertical resolution improves forecast skill.

## Model Structure

To make the comparison fair, we keep the core of the model the same and only adjust the channel axis which is responsible
for encoding the pressure levels. This means we only change two layers:

- `patch_embedding` (in_channels → embed) and
- `patch_recovery` (embed → out_channels).

**Everything else — the entire SWIN core (downsample, transformer blocks, upsample, mix) — runs in `embed_dim` space and 
is architecturally identical** between the 13- and 37-level models. Therefore, `embedding_dimension`, `n_attn_blocks`, 
`n_attn_heads`, `patch_size`, and the window sizes are the same accross both models.

## Planned Experiments

1. **Compare 13 vs 37** — Compare total predictive performance of 37-level and 13-level model.
2. **Subset-evaluation** — score the 37-level model on *exactly* the 13 standard
   levels vs the 13-level model.
3. **Frozen-core transfer** — freeze a trained core, retrain *only* the two I/O
   convs for the other level count. Isolates whether benefit lives in the
   representation or the I/O.