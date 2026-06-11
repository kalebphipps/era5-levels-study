"""Coarsen 0.25deg constant masks to the SAME 120x240 grid as coarsen_to_1p5.py.

Your pre-made 1.5deg masks (*_121_240.npy) are on the standard 121x240 grid,
which does NOT match the factor-6 block-mean output (120x240). To keep masks and
data on an identical grid, block-mean the 0.25deg masks the same way the data is
coarsened (drop the trailing odd latitude, then average 6x6 blocks).

    python scripts/coarsen_masks.py --factor 6 --out-dir data/constant_masks_1p5 \
        data/constant_masks/soil_type_normalized.npy \
        data/constant_masks/topography_normalized.npy \
        data/constant_masks/land_mask.npy

Output keeps the same filenames, so the model config's `constant_masks` list is
unchanged; only `constant_masks_path` points at the 1.5deg directory.
"""

import argparse
import os

import numpy as np


def coarsen_2d(arr: np.ndarray, factor: int) -> np.ndarray:
    """Block-mean a (lat, lon) array by `factor`, trimming any remainder."""
    a = arr.astype(np.float64)
    nlat = (a.shape[0] // factor) * factor
    nlon = (a.shape[1] // factor) * factor
    a = a[:nlat, :nlon]
    a = a.reshape(nlat // factor, factor, nlon // factor, factor).mean(axis=(1, 3))
    return a.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("masks", nargs="+", help="0.25deg mask .npy files (lat, lon)")
    ap.add_argument("--factor", type=int, default=6)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for path in args.masks:
        arr = np.load(path)
        if arr.ndim != 2:
            print(f"skip {path}: expected 2D (lat, lon), got {arr.shape}")
            continue
        out = coarsen_2d(arr, args.factor)
        dst = os.path.join(args.out_dir, os.path.basename(path))
        np.save(dst, out)
        print(f"{os.path.basename(path)}: {arr.shape} -> {out.shape}  {dst}")


if __name__ == "__main__":
    main()
