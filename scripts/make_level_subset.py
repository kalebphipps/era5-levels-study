"""Write a 13-level subset zarr (+ subset normalization) from a 37-level zarr.

The beast dataset loads ALL features present in a zarr, so the 13-level run
needs its own store containing exactly the 84 features (6 surface + 6 pressure x
13), in build_ordered_variables(13) order. This selects them by position from
the 37-level store, and (optionally) subsets the matching norm_mean/std .npy so
you don't recompute normalization.

RESUMABLE + writes zarr v2 (same as coarsen_to_1p5.py): it reads the whole
37-level store (~9 TB for 40 years hourly), so run it as a SLURM job, not on the
login node. Re-running with the same args continues from however many timesteps
are already written.

    python scripts/make_level_subset.py \
        --in  $WS/data/era5_37level_1p5.zarr \
        --out $WS/data/era5_13level_1p5.zarr \
        --norm-in data/normalization_1p5_37 \
        --norm-out data/normalization_1p5_13

The 13 levels are scattered within the 37-level channel layout (each variable
keeps only 13 of its 37 levels), so the selection is by computed index, not a
contiguous slice.
"""

import argparse
import os

import numpy as np
import xarray as xr

from era5_levels.variable_layout import (
    PRESSURE_LEVELS_13,
    PRESSURE_LEVELS_37,
    build_ordered_variables,
)


def subset_indices() -> list[int]:
    """Return the 37-level channel indices that make up the 13-level layout.

    Returns
    -------
    list of int
        Channel positions in the 37-level store, in
        ``build_ordered_variables(PRESSURE_LEVELS_13)`` order.
    """
    full = build_ordered_variables(PRESSURE_LEVELS_37)
    sub = build_ordered_variables(PRESSURE_LEVELS_13)
    pos = {name: i for i, name in enumerate(full)}
    return [pos[name] for name in sub]


def main():
    """Parse CLI args and write the 13-level subset zarr (+ norm), resumably."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="37-level zarr")
    ap.add_argument("--out", required=True, help="output 13-level zarr")
    ap.add_argument("--norm-in", help="dir with 37-level norm_mean.npy / norm_std.npy")
    ap.add_argument("--norm-out", help="dir to write the 13-level norm .npy")
    ap.add_argument("--lat", default="latitude")
    ap.add_argument("--lon", default="longitude")
    ap.add_argument("--var", default="fields")
    ap.add_argument("--block", type=int, default=48,
                    help="timesteps per append = checkpoint granularity")
    args = ap.parse_args()

    idx = subset_indices()
    assert len(idx) == 84, f"expected 84 features, got {len(idx)}"

    # Normalization subset is cheap -> do it first (idempotent).
    if args.norm_in and args.norm_out:
        os.makedirs(args.norm_out, exist_ok=True)
        for name in ("norm_mean.npy", "norm_std.npy"):
            arr = np.load(os.path.join(args.norm_in, name))
            np.save(os.path.join(args.norm_out, name), arr[idx])
        print(f"wrote subset normalization (84 features) -> {args.norm_out}")

    ds = xr.open_zarr(args.inp)
    n_total = ds.sizes["time"]
    nfeat = len(idx)
    nlat, nlon = ds.sizes[args.lat], ds.sizes[args.lon]
    print(f"input: {dict(ds.sizes)}  ({n_total} timesteps) -> 84-feature subset", flush=True)

    # Resume: how many timesteps already written?
    done = 0
    if os.path.exists(args.out):
        try:
            done = xr.open_zarr(args.out, consolidated=False).sizes["time"]
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: could not read existing output ({e}); starting fresh")
            done = 0
    if done >= n_total:
        print(f"already complete: {done}/{n_total} timesteps in {args.out}")
        return
    print(f"resuming from timestep {done}/{n_total}", flush=True)

    enc = {args.var: {"chunks": (1, nfeat, nlat, nlon)}}
    first = done == 0
    for start in range(done, n_total, args.block):
        stop = min(start + args.block, n_total)
        blk = ds.isel(time=slice(start, stop), feature=idx)
        blk = blk.chunk({"time": 1, "feature": nfeat, args.lat: nlat, args.lon: nlon})
        # Drop inherited encoding (stale chunks + v2 Blosc codec) and write zarr v2.
        for name in blk.variables:
            blk[name].encoding.clear()
        if first:
            blk.to_zarr(args.out, mode="w", zarr_format=2, consolidated=True, encoding=enc)
            first = False
        else:
            blk.to_zarr(args.out, mode="a", append_dim="time", zarr_format=2,
                        consolidated=True)
        print(f"  wrote {stop}/{n_total}", flush=True)

    print(f"output: ({n_total}, {nfeat}, {nlat}, {nlon}) -> {args.out}")


if __name__ == "__main__":
    main()
