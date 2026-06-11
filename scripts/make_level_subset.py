"""Write a 13-level subset zarr (+ subset normalization) from a 37-level zarr.

The beast dataset loads ALL features present in a zarr, so the 13-level run
needs its own store containing exactly the 84 features (6 surface + 6 pressure x
13), in build_ordered_variables(13) order. This selects them by position from
the 37-level store, and (optionally) subsets the matching norm_mean/std .npy so
you don't recompute normalization.

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
    full = build_ordered_variables(PRESSURE_LEVELS_37)
    sub = build_ordered_variables(PRESSURE_LEVELS_13)
    pos = {name: i for i, name in enumerate(full)}
    return [pos[name] for name in sub]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="37-level zarr")
    ap.add_argument("--out", required=True, help="output 13-level zarr")
    ap.add_argument("--norm-in", help="dir with 37-level norm_mean.npy / norm_std.npy")
    ap.add_argument("--norm-out", help="dir to write the 13-level norm .npy")
    args = ap.parse_args()

    idx = subset_indices()
    assert len(idx) == 84, f"expected 84 features, got {len(idx)}"

    ds = xr.open_zarr(args.inp)
    ds_sub = ds.isel(feature=idx)
    ds_sub.to_zarr(args.out, mode="w", consolidated=True)
    print(f"wrote {args.out}: {dict(ds_sub.sizes)}")

    if args.norm_in and args.norm_out:
        os.makedirs(args.norm_out, exist_ok=True)
        for name in ("norm_mean.npy", "norm_std.npy"):
            arr = np.load(os.path.join(args.norm_in, name))
            np.save(os.path.join(args.norm_out, name), arr[idx])
        print(f"wrote subset normalization -> {args.norm_out}")


if __name__ == "__main__":
    main()
