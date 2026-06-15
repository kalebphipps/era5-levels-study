"""Coarsen a 0.25-degree stacked ERA5 zarr to ~1.5 degrees (factor-6 block mean).

RESUMABLE + chainable: writes the output in time-blocks, appending each block.
If the job dies, just re-run with the SAME arguments — it reads how many
timesteps are already in the output and continues from there. So you can chain
it with afterany jobs (slurm/submit_coarsen_chain.sh) until it finishes.

WeatherBench-2's ready-made 1.5deg ERA5 has only 13 pressure levels, so to get
1.5deg WITH the full 37 levels we coarsen YOUR 0.25deg 37-level zarr. The
(time, feature, lat, lon) layout and the 228-channel feature order are preserved
exactly, and 0.25deg + 1.5deg then share identical source data.

    python scripts/coarsen_to_1p5.py --in IN.zarr --out OUT.zarr \
        --factor 6 --time-range 1990-01-01 2022-12-31 --time-stride 6

0.25deg (721x1440) --factor 6 + boundary="trim"--> 120x240 (the trailing odd
latitude is dropped, which the dataset would clip anyway). Coarsen the constant
masks the SAME way with scripts/coarsen_masks.py so they share this 120x240 grid.

Notes
-----
- Streams via dask; reading the 0.25deg input is the cost (~0.95 GB/timestep).
  Use --time-stride 6 on hourly data (-> 6-hourly, matches dt=6 training).
- Recompute normalization on the OUTPUT (coarsening shrinks per-feature std).
- sea_surface_temperature was stored with land=0, so its block mean is slightly
  biased near coasts. Fine for a poster.
"""

import argparse
import os

import xarray as xr


def main():
    """Parse CLI args and coarsen the input zarr block-by-block (resumable)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="0.25deg 37-level zarr")
    ap.add_argument("--out", required=True, help="output 1.5deg zarr")
    ap.add_argument("--factor", type=int, default=6, help="coarsen factor (6 = 0.25->1.5)")
    ap.add_argument("--lat", default="latitude")
    ap.add_argument("--lon", default="longitude")
    ap.add_argument("--var", default="fields")
    ap.add_argument("--time-range", nargs=2, metavar=("START", "END"), default=None)
    ap.add_argument("--time-stride", type=int, default=1,
                    help="keep every Nth timestep. Hourly source -> 6 gives 6-hourly "
                         "(matches dt=6) and cuts reads ~6x. Use 1 if already 6-hourly.")
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

    # coarse dims (lazy metadata only)
    meta = ds.isel(time=slice(0, 1)).coarsen(
        {args.lat: args.factor, args.lon: args.factor}, boundary="trim").mean()
    nfeat = meta.sizes["feature"]
    nlat, nlon = meta.sizes[args.lat], meta.sizes[args.lon]

    # how many timesteps already written? -> resume point
    done = 0
    if os.path.exists(args.out):
        try:
            # consolidated=False: read the store's true current state (a
            # killed link may not have re-consolidated), and skip the slow
            # consolidated-metadata fallback warning.
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
        # Drop encoding inherited from the source: it carries stale chunk shapes
        # (the 0.25deg grid) and a numcodecs Blosc compressor that zarr-python 3.x
        # refuses to write into a v3 array. We write zarr v2 (matching the source)
        # with fresh chunking -- same approach as era5_parallel_processing.
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
