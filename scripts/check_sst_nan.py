"""Check whether ``sea_surface_temperature`` is NaN over the OCEAN (a data bug).

SST is *legitimately* NaN over land (~29% of the globe) and finite over the ocean
(~71%). If a stacked ERA5 zarr has SST that is (near-)entirely NaN, then it is
NaN over the sea too — the bug this script tests for.

Runs on any of the stacked ``(time, feature, lat, lon)`` zarrs (the 0.25° original
or the coarsened 1.5° / 13-level subset). No land mask is needed: it reports the
NaN fraction directly and compares it against the ~29% land baseline, and
cross-checks against ``2m_temperature`` (which should be finite everywhere).

Examples
--------
    python scripts/check_sst_nan.py \\
        /path/to/era5_37level_0p25.zarr \\
        $WS/data/era5_37level_1p5.zarr \\
        $WS/data/era5_13level_1p5.zarr
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import xarray as xr

# Fraction of Earth's surface that is land, where ERA5 SST is legitimately NaN.
LAND_FRACTION = 0.29

# Positional fallback indices within the surface block (build_ordered_variables
# order: mslp, 10u, 10v, 2t, sst, tp), used when the `feature` coordinate has no
# string labels to select by name.
SST_FALLBACK_INDEX = 4
REF_FALLBACK_INDEX = 3  # 2m_temperature


def _main_var(ds: xr.Dataset, feature: str, requested: str) -> str:
    """Return the name of the stacked data variable that carries the feature dim.

    Parameters
    ----------
    ds : xarray.Dataset
        The opened zarr dataset.
    feature : str
        Name of the feature (channel) dimension.
    requested : str
        The variable name the user asked for (used if present).

    Returns
    -------
    str
        A data-variable name that has ``feature`` as a dimension.

    Raises
    ------
    KeyError
        If no data variable uses the feature dimension.
    """
    if requested in ds and feature in ds[requested].dims:
        return requested
    for name, da in ds.data_vars.items():
        if feature in da.dims:
            return name
    raise KeyError(
        f"No data variable with a '{feature}' dimension in {list(ds.data_vars)}."
    )


def _select_feature(da: xr.DataArray, feature: str, name: str,
                    fallback_index: int) -> xr.DataArray:
    """Select one feature by label, falling back to a positional index.

    Parameters
    ----------
    da : xarray.DataArray
        The stacked ``(time, feature, lat, lon)`` array.
    feature : str
        Name of the feature (channel) dimension.
    name : str
        Feature label to select (e.g. ``"sea_surface_temperature"``).
    fallback_index : int
        Index along ``feature`` to use if the dimension has no string labels.

    Returns
    -------
    xarray.DataArray
        The selected single-feature slice.
    """
    if feature in da.coords:
        coord = da[feature].values
        if coord.dtype.kind in ("U", "S", "O"):  # string labels present
            labels = [str(v) for v in coord]
            if name in labels:
                return da.isel({feature: labels.index(name)})
    return da.isel({feature: fallback_index})


def check(path: str, var: str = "fields", feature: str = "feature",
          time: str = "time", n_times: int = 3) -> None:
    """Report SST NaN statistics for one zarr and print a verdict.

    Parameters
    ----------
    path : str
        Path to a stacked ERA5 zarr.
    var : str, optional
        Name of the stacked data variable (auto-detected if absent).
    feature : str, optional
        Name of the feature (channel) dimension.
    time : str, optional
        Name of the time dimension.
    n_times : int, optional
        Number of timesteps to sample (first / middle / last).
    """
    ds = xr.open_zarr(path)
    var = _main_var(ds, feature, var)
    da = ds[var]

    nt = da.sizes[time]
    idxs = sorted({0, nt // 2, nt - 1})[:n_times]
    sst = _select_feature(da, feature, "sea_surface_temperature",
                          SST_FALLBACK_INDEX).isel({time: idxs}).compute()
    ref = _select_feature(da, feature, "2m_temperature",
                          REF_FALLBACK_INDEX).isel({time: idxs}).compute()

    total = int(sst.size)
    sst_vals = sst.values
    sst_nan_frac = float(np.isnan(sst_vals).sum()) / total
    ref_nan_frac = float(np.isnan(ref.values).sum()) / total
    finite = sst_vals[np.isfinite(sst_vals)]

    print(f"\n=== {path} ===")
    print(f"  data var '{var}', grid {dict(sst.sizes)}, sampled timesteps {idxs}")
    print(f"  SST   NaN: {sst_nan_frac:7.2%}   finite cells: {finite.size}/{total}")
    if finite.size:
        print(f"        finite min/mean/max: "
              f"{finite.min():.3f} / {finite.mean():.3f} / {finite.max():.3f}")
    print(f"  2m_t  NaN: {ref_nan_frac:7.2%}   (reference — should be ~0%)")

    over_sea = sst_nan_frac - LAND_FRACTION
    if sst_nan_frac >= 0.99:
        print("  VERDICT: BROKEN — SST is entirely NaN (NaN over the sea too).")
    elif over_sea > 0.05:
        print(f"  VERDICT: SUSPECT — ~{over_sea:.0%} of cells are NaN beyond the "
              f"~{LAND_FRACTION:.0%} land baseline, i.e. NaN over the sea.")
    elif sst_nan_frac < 0.01:
        print("  VERDICT: NO NaN — SST has no NaN at all (land is filled, not "
              "masked); no sea-NaN bug, but confirm land handling is intended.")
    else:
        print("  VERDICT: OK — NaN fraction consistent with land-only masking.")


def main() -> None:
    """Parse CLI args and run the SST NaN check on each given zarr."""
    ap = argparse.ArgumentParser(
        description="Test whether sea_surface_temperature is NaN over the ocean.")
    ap.add_argument("paths", nargs="+", help="stacked ERA5 zarr(s) to check")
    ap.add_argument("--var", default="fields", help="stacked data variable name")
    ap.add_argument("--feature", default="feature", help="feature/channel dim name")
    ap.add_argument("--time", default="time", help="time dim name")
    ap.add_argument("--n-times", type=int, default=3, help="timesteps to sample")
    args = ap.parse_args()

    for p in args.paths:
        try:
            check(p, var=args.var, feature=args.feature, time=args.time,
                  n_times=args.n_times)
        except Exception as e:  # noqa: BLE001 - diagnostic: report and continue
            print(f"\n=== {p} ===\n  ERROR: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
