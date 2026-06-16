"""Coarsen 0.25deg constant masks to the same grid as the 1.5 degree data."""

import argparse
import os

import numpy as np


def coarsen_2d(arr: np.ndarray, factor: int) -> np.ndarray:
    """Coarsen a 2D ``(lat, lon)`` constant mask, trimming any remainder.

    Parameters
    ----------
    arr : numpy.ndarray
        Input field of shape ``(lat, lon)``.
    factor : int
        Block size; each ``factor x factor`` block is averaged. Trailing rows/
        columns that don't fill a block are dropped.

    Returns
    -------
    numpy.ndarray
        Coarsened ``float32`` field of shape ``(lat // factor, lon // factor)``.
    """
    a = arr.astype(np.float64)
    nlat = (a.shape[0] // factor) * factor
    nlon = (a.shape[1] // factor) * factor
    a = a[:nlat, :nlon]
    a = a.reshape(nlat // factor, factor, nlon // factor, factor).mean(axis=(1, 3))
    return a.astype(np.float32)


def main():
    """Parse args and coarsen each mask."""
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
