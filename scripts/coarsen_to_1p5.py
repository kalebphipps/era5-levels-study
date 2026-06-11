"""Coarsen a 0.25-degree stacked ERA5 zarr to ~1.5 degrees (factor-6 block mean).

WeatherBench-2's ready-made 1.5deg ERA5 has only 13 pressure levels, so to get
1.5deg WITH the full 37 levels we coarsen YOUR existing 0.25deg 37-level zarr.
Bonus: 1.5deg and 0.25deg then share identical source data, so the
resolution comparison is clean. The (time, feature, lat, lon) layout and the
228-channel feature order are preserved exactly.

    python scripts/coarsen_to_1p5.py \
        --in  /path/era5_37level_0p25.zarr \
        --out $WS/data/era5_37level_1p5.zarr \
        --factor 6 --time-range 2010-01-01 2021-12-31

0.25deg (721x1440) --factor 6--> 120x240 (~1.5deg). boundary="trim" drops the
trailing odd latitude (721) which the dataset would clip anyway.

Notes
-----
- This streams via dask; on a big multi-decade store it is I/O heavy. Restrict
  --time-range (e.g. ~10 years train + 1-2 valid) to keep it fast for the poster.
- sea_surface_temperature was stored with land = 0 in the stacked data, so the
  block mean slightly biases coastal SST. Fine for a poster; flag if you care.
- Recompute normalization on the OUTPUT (era5_levels.data.calculate_normalization
  / your norm script) — coarsening changes the per-feature std.
"""

import argparse

import xarray as xr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="0.25deg 37-level zarr")
    ap.add_argument("--out", required=True, help="output 1.5deg zarr")
    ap.add_argument("--factor", type=int, default=6, help="coarsen factor (6 = 0.25->1.5)")
    ap.add_argument("--lat", default="latitude")
    ap.add_argument("--lon", default="longitude")
    ap.add_argument("--var", default="fields")
    ap.add_argument("--time-range", nargs=2, metavar=("START", "END"), default=None)
    args = ap.parse_args()

    ds = xr.open_zarr(args.inp)
    if args.time_range:
        ds = ds.sel(time=slice(args.time_range[0], args.time_range[1]))
    print(f"input:  {dict(ds.sizes)}")

    ds_c = ds.coarsen(
        {args.lat: args.factor, args.lon: args.factor}, boundary="trim"
    ).mean()

    nfeat = ds_c.sizes["feature"]
    nlat = ds_c.sizes[args.lat]
    nlon = ds_c.sizes[args.lon]
    ds_c = ds_c.chunk({"time": 1, "feature": nfeat, args.lat: nlat, args.lon: nlon})

    enc = {args.var: {"chunks": (1, nfeat, nlat, nlon)}}
    ds_c.to_zarr(args.out, mode="w", consolidated=True, encoding=enc)
    print(f"output: {dict(ds_c.sizes)} -> {args.out}")


if __name__ == "__main__":
    main()
