"""Coarsen a 0.25-degree stacked ERA5 zarr to ~1.5 degrees."""

import argparse
import os

import xarray as xr


def main():
    """Parse args and coarsen the input zarr block-by-block."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="0.25deg 37-level zarr")
    ap.add_argument("--out", required=True, help="output 1.5deg zarr")
    ap.add_argument("--factor", type=int, default=6, help="coarsen factor (6 = 0.25->1.5)")
    ap.add_argument("--lat", default="latitude")
    ap.add_argument("--lon", default="longitude")
    ap.add_argument("--var", default="fields")
    ap.add_argument("--time-range", nargs=2, metavar=("START", "END"), default=None)
    ap.add_argument("--time-stride", type=int, default=1,
                    help="keep every Nth timestep.")
    ap.add_argument("--block", type=int, default=48,
                    help="timesteps written per append = checkpoint granularity")
    args = ap.parse_args()

    ds = xr.open_zarr(args.inp)
    if args.time_range:
        ds = ds.sel(time=slice(args.time_range[0], args.time_range[1]))
    if args.time_stride > 1:
        ds = ds.isel(time=slice(None, None, args.time_stride))
    n_total = ds.sizes["time"]
    print(f"input selection: {dict(ds.sizes)}  ({n_total} timesteps)", flush=True)

    # Coarsen dimensions
    meta = ds.isel(time=slice(0, 1)).coarsen(
        {args.lat: args.factor, args.lon: args.factor}, boundary="trim").mean()
    nfeat = meta.sizes["feature"]
    nlat, nlon = meta.sizes[args.lat], meta.sizes[args.lon]

    # Resume point.
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
    print(f"resuming from timestep {done}/{n_total} "
          f"(output grid feature={nfeat} lat={nlat} lon={nlon})", flush=True)

    enc = {args.var: {"chunks": (1, nfeat, nlat, nlon)}}
    first = done == 0
    for start in range(done, n_total, args.block):
        stop = min(start + args.block, n_total)
        blk = ds.isel(time=slice(start, stop)).coarsen(
            {args.lat: args.factor, args.lon: args.factor}, boundary="trim").mean()
        blk = blk.chunk({"time": 1, "feature": nfeat, args.lat: nlat, args.lon: nlon})
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
