"""Offline poster figures from the study outputs.

Runs on a laptop (matplotlib + pandas + numpy) — no beast / GPU. Reads the
`metrics.csv` files written during validation and the `.npy` map dumps, and
produces the poster figures:

  1. per-level RMSE curves        (13 vs 37, with persistence/climatology baselines)
  2. 13-vs-37 improvement heatmap (variable x level, % RMSE change)
  3. learning curves              (mean RMSE vs epoch)
  4. forecast/error maps          (pred | truth | error) from the .npy dumps

Examples
--------
    # both runs -> curves + heatmap + per-run learning curves
    python scripts/plot_results.py \
        --csv13 $WS/results/levels13/metrics.csv \
        --csv37 $WS/results/levels37/metrics.csv \
        --maps-dir $WS/results/levels37/maps/epoch_0 \
        --out figures/

    # single run is fine too (curves + learning curve + baselines, no heatmap)
    python scripts/plot_results.py --csv37 $WS/results/levels37/metrics.csv --out figures/
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Pressure variables we usually show; surface vars are handled separately.
DEFAULT_PRESSURE_VARS = ["geopotential", "temperature", "u_component_of_wind",
                         "specific_humidity"]


def parse_variable(name: str):
    """'geopotential_500' -> ('geopotential', 500); '2m_temperature' -> (name, None)."""
    head, _, tail = name.rpartition("_")
    if head and tail.isdigit():
        return head, int(tail)
    return name, None


def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df[["base", "level"]] = df["variable"].apply(
        lambda v: pd.Series(parse_variable(v)))
    return df


def latest_epoch(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["epoch"] == df["epoch"].max()].copy()


# ---------------------------------------------------------------------------- #
# 1. per-level RMSE curves
# ---------------------------------------------------------------------------- #
def plot_per_level_curves(df13, df37, variables, out_dir):
    runs = [("37-level", df37, "C1"), ("13-level", df13, "C0")]
    runs = [(lbl, d, c) for lbl, d, c in runs if d is not None]
    if not runs:
        return
    n = len(variables)
    ncol = 2
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.2 * nrow), squeeze=False)
    for ax, var in zip(axes.flat, variables):
        for lbl, d, c in runs:
            sub = latest_epoch(d)
            sub = sub[(sub["base"] == var) & sub["level"].notna()].sort_values("level")
            if sub.empty:
                continue
            ax.plot(sub["model"], sub["level"], "-o", color=c, label=f"{lbl} model", ms=3)
            if "persistence" in sub:
                ax.plot(sub["persistence"], sub["level"], ":", color=c, alpha=0.5,
                        label=f"{lbl} persistence")
        ax.set_title(var)
        ax.set_xlabel("latitude-weighted RMSE")
        ax.set_ylabel("pressure level [hPa]")
        ax.invert_yaxis()  # surface (1000) at bottom
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    p = os.path.join(out_dir, "per_level_rmse.png")
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


# ---------------------------------------------------------------------------- #
# 2. improvement heatmap (37 vs 13)
# ---------------------------------------------------------------------------- #
def plot_improvement_heatmap(df13, df37, out_dir):
    if df13 is None or df37 is None:
        return
    a = latest_epoch(df13).set_index("variable")["model"]
    b = latest_epoch(df37).set_index("variable")["model"]
    common = a.index.intersection(b.index)
    rows = {}
    for v in common:
        base, lvl = parse_variable(v)
        if lvl is None:
            continue
        pct = 100.0 * (b[v] - a[v]) / a[v]   # <0 => 37 better
        rows.setdefault(base, {})[lvl] = pct
    if not rows:
        return
    bases = sorted(rows)
    levels = sorted({lvl for r in rows.values() for lvl in r}, reverse=True)
    grid = np.full((len(bases), len(levels)), np.nan)
    for i, base in enumerate(bases):
        for j, lvl in enumerate(levels):
            if lvl in rows[base]:
                grid[i, j] = rows[base][lvl]

    vmax = np.nanmax(np.abs(grid))
    fig, ax = plt.subplots(figsize=(0.5 * len(levels) + 3, 0.5 * len(bases) + 2))
    im = ax.imshow(grid, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(levels)), levels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(bases)), bases, fontsize=8)
    ax.set_xlabel("pressure level [hPa]")
    ax.set_title("RMSE change: 37-level vs 13-level  [%]  (blue = 37 better)")
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    p = os.path.join(out_dir, "improvement_heatmap.png")
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


# ---------------------------------------------------------------------------- #
# 3. learning curves
# ---------------------------------------------------------------------------- #
def plot_learning_curves(dfs, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    for lbl, d, c in dfs:
        if d is None:
            continue
        g = d.groupby("epoch")[["model"]].mean().reset_index()
        ax.plot(g["epoch"], g["model"], "-o", color=c, label=f"{lbl} model", ms=3)
        for base_metric, style in (("persistence", ":"), ("climatology", "--")):
            if base_metric in d:
                gb = d.groupby("epoch")[[base_metric]].mean().reset_index()
                ax.plot(gb["epoch"], gb[base_metric], style, color=c, alpha=0.5,
                        label=f"{lbl} {base_metric}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean latitude-weighted RMSE")
    ax.set_title("learning curves")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(out_dir, "learning_curves.png")
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


# ---------------------------------------------------------------------------- #
# 4. maps
# ---------------------------------------------------------------------------- #
def plot_maps(maps_dir, out_dir):
    preds = sorted(glob.glob(os.path.join(maps_dir, "*_pred.npy")))
    for pred_path in preds:
        var = os.path.basename(pred_path)[: -len("_pred.npy")]
        pred = np.load(pred_path)
        true = np.load(os.path.join(maps_dir, f"{var}_true.npy"))
        err = np.load(os.path.join(maps_dir, f"{var}_err.npy"))
        vmin, vmax = float(min(pred.min(), true.min())), float(max(pred.max(), true.max()))
        emax = float(np.abs(err).max())
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, data, title, kw in (
            (axes[0], pred, f"prediction: {var}", dict(vmin=vmin, vmax=vmax, cmap="viridis")),
            (axes[1], true, f"ERA5 truth: {var}", dict(vmin=vmin, vmax=vmax, cmap="viridis")),
            (axes[2], err, f"error (pred-truth): {var}", dict(vmin=-emax, vmax=emax, cmap="RdBu_r")),
        ):
            im = ax.imshow(data, origin="upper", **kw)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        fig.tight_layout()
        p = os.path.join(out_dir, f"map_{var}.png")
        fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print("wrote", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv13", help="metrics.csv from the 13-level run")
    ap.add_argument("--csv37", help="metrics.csv from the 37-level run")
    ap.add_argument("--maps-dir", help="dir with <var>_{pred,true,err}.npy")
    ap.add_argument("--vars", nargs="*", default=DEFAULT_PRESSURE_VARS,
                    help="pressure variables for the per-level curves")
    ap.add_argument("--out", default="figures", help="output dir for PNGs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df13 = load(args.csv13) if args.csv13 else None
    df37 = load(args.csv37) if args.csv37 else None

    if df13 is not None or df37 is not None:
        plot_per_level_curves(df13, df37, args.vars, args.out)
        plot_learning_curves(
            [("13-level", df13, "C0"), ("37-level", df37, "C1")], args.out)
    plot_improvement_heatmap(df13, df37, args.out)
    if args.maps_dir:
        plot_maps(args.maps_dir, args.out)


if __name__ == "__main__":
    main()
