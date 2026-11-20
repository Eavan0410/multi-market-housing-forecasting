"""Compare pooled AR coefficients with Transformer gradients and Integrated Gradients.

Lag 1 is the latest input quarter. Absolute attribution shares are calculated per
seed before averaging. Retrained forecasts are checked against the saved main run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_backtest import load_samples  # noqa: E402

from housing_forecast.backtesting.runner import pool_national, training_samples  # noqa: E402
from housing_forecast.backtesting.scaling import WindowStandardizer  # noqa: E402
from housing_forecast.backtesting.splits import (  # noqa: E402
    check_no_leakage, end_series, eval_mask, last_label_origin, make_splits,
)
from housing_forecast.config import load_config  # noqa: E402
from housing_forecast.evaluation.explain import ar_attributions, ar_lag_weights, transformer_attributions  # noqa: E402
from housing_forecast.models.base import set_global_seed  # noqa: E402
from housing_forecast.models.baselines import ARPooled, ARPooled8  # noqa: E402

BANDS = {"h3-6": range(3, 7), "h16-20": range(16, 21)}
NAMES = {"transformer": "Shared Transformer", "ar4": "Pooled AR(4)", "ar8": "Pooled AR(8)"}


def sha256(path: str) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def abs_share(ig: np.ndarray) -> np.ndarray:
    """(n, H, L) share of |contribution| per lag within each forecast and horizon."""
    a = np.abs(ig)
    total = a.sum(axis=2, keepdims=True)
    return a / np.where(total == 0, np.nan, total)


def reproduction_gap(reference: pd.DataFrame | None, test, seed: int, pred: np.ndarray, split_id: str,
                     tol: float) -> float:
    """Largest |retrained - saved| forecast. Stops if a labelled forecast is missing or the gap exceeds tol."""
    if reference is None:
        return float("nan")
    ref = reference[reference["seed"].eq(seed)]
    if ref.duplicated(["series_id", "origin_quarter", "horizon"]).any():
        raise SystemExit(f"{split_id} seed {seed}: duplicate rows in the saved forecasts")
    table = ref.pivot(index=["series_id", "origin_quarter"], columns="horizon", values="y_pred")
    rows = pd.MultiIndex.from_arrays([test.meta["series_id"].to_numpy(), test.meta["origin"].astype(str).to_numpy()])
    saved = table.reindex(index=rows, columns=list(test.horizons)).to_numpy()
    labelled = ~np.isnan(test.y)
    missing = labelled & np.isnan(saved)
    if missing.any():
        raise SystemExit(f"{split_id} seed {seed}: {int(missing.sum())} labelled forecasts are missing from the "
                         "saved file, so the retrained model cannot be checked against it")
    gap = float(np.abs(saved[labelled] - pred[labelled]).max())
    if gap > tol:
        raise SystemExit(f"{split_id} seed {seed}: retrained forecasts differ from the saved ones by {gap:.2e} "
                         f"(> {tol:.0e}). Run with OMP_NUM_THREADS=1 on the same data and code; no results written.")
    return gap


def run(cfg, samples, splits, steps: int, reference: pd.DataFrame | None, tol: float):
    from housing_forecast.models.transformer import TransformerForecaster

    j = samples.feature_names.index("hpi_qoq")
    seeds = cfg["run"]["seeds"]
    gap = cfg["periods"]["gap_quarters"]
    meta, checks = [], []
    arrays = {k: [] for k in ("tr_grad", "tr_ig", "tr_share", "tr_pred", "ar4_ig", "ar8_ig", "y", "x_window")}
    ar_weights = {"ar4": [], "ar8": [], "split_id": []}
    for fold, split in enumerate(splits):
        train = training_samples(samples, split, pool_national())
        test = samples.subset(eval_mask(samples, split, cfg["target"]["eval_series"]))
        if len(test) == 0:
            continue
        check_no_leakage(train, test, split, gap)
        scaler = WindowStandardizer().fit(train.X)
        baseline = float(scaler.mean_[j])                  # same "average input" for every model
        ar_weights["split_id"].append(split.split_id)

        for key, factory in (("ar4", ARPooled), ("ar8", ARPooled8)):
            ar = factory()
            ar.fit(train, 0)
            ar_weights[key].append((len(test), ar_lag_weights(ar, test.X.shape[1])))
            arrays[f"{key}_ig"].append(ar_attributions(ar, test.X, j, baseline))

        train_s = train.replace(X=scaler.transform(train.X))
        grad, ig, share, pred = [], [], [], []
        for seed in seeds:
            set_global_seed(seed)
            model = TransformerForecaster()
            model.fit(train_s, seed)
            att = transformer_attributions(model, scaler, test.X, feature=j, steps=steps)
            grad.append(att["grad"]), ig.append(att["ig"]), share.append(abs_share(att["ig"]))
            pred.append(att["pred"])
            diff = reproduction_gap(reference, test, seed, att["pred"], split.split_id, tol)
            checks.append({"split_id": split.split_id, "seed": seed, "max_abs_diff_vs_saved": diff,
                           "ig_completeness_error": att["completeness_error"]})
            print(f"{split.split_id} seed {seed}: |diff vs saved| {diff:.2e}, "
                  f"IG completeness error {att['completeness_error']:.2e}", flush=True)
        arrays["tr_grad"].append(np.mean(grad, axis=0))
        arrays["tr_ig"].append(np.mean(ig, axis=0))          # signed, for the seed-averaged forecast
        arrays["tr_share"].append(np.mean(share, axis=0))    # |contribution| share, per seed then averaged
        arrays["tr_pred"].append(np.mean(pred, axis=0))
        arrays["y"].append(test.y)
        arrays["x_window"].append(test.X[:, :, j])          # raw input window, oldest quarter first
        meta.append(pd.DataFrame({"split_id": split.split_id, "series_id": test.meta["series_id"].to_numpy(),
                                  "origin_quarter": test.meta["origin"].astype(str).to_numpy()}))
    return (pd.concat(meta, ignore_index=True), {k: np.concatenate(v) for k, v in arrays.items()},
            ar_weights, pd.DataFrame(checks))


def summarize(meta, arrays, ar_weights, horizons) -> pd.DataFrame:
    L = arrays["tr_grad"].shape[2]
    groups = {f"h{h}": [h] for h in horizons} | {k: list(v) for k, v in BANDS.items()}
    hidx = {h: i for i, h in enumerate(horizons)}

    rows = []
    for band, hs in groups.items():
        ix = [hidx[h] for h in hs]
        g = arrays["tr_grad"][:, ix, :].mean(axis=1)
        s = np.nanmean(arrays["tr_share"][:, ix, :], axis=1)
        for lag in range(L):
            rows.append({"model": NAMES["transformer"], "horizon": band, "lag": lag + 1,
                         "weight_mean": np.mean(g[:, lag]), "weight_p10": np.percentile(g[:, lag], 10),
                         "weight_p90": np.percentile(g[:, lag], 90),
                         "contribution_share": np.nanmean(s[:, lag])})
        for key in ("ar4", "ar8"):
            n = np.array([c for c, _ in ar_weights[key]], dtype=float)
            W = np.stack([w for _, w in ar_weights[key]])[:, ix, :].mean(axis=1)
            sa = np.nanmean(abs_share(arrays[f"{key}_ig"])[:, ix, :], axis=1)
            for lag in range(L):
                rows.append({"model": NAMES[key], "horizon": band, "lag": lag + 1,
                             "weight_mean": float(np.average(W[:, lag], weights=n)),
                             "weight_p10": float(W[:, lag].min()), "weight_p90": float(W[:, lag].max()),
                             "contribution_share": np.nanmean(sa[:, lag])})
    return pd.DataFrame(rows)


def plot(table: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = ["h1", "h3-6", "h16-20"]
    styles = {"Shared Transformer": ("#1f4e79", "o", "-"), "Pooled AR(4)": ("#e69138", "v", "--"),
              "Pooled AR(8)": ("#b45f06", "^", ":")}
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for c, band in enumerate(panels):
        sub = table[table["horizon"].eq(band)]
        for name, g in sub.groupby("model"):
            color, marker, ls = styles[name]
            axes[0, c].plot(g["lag"], g["weight_mean"], color=color, marker=marker, ls=ls, label=name)
            if name == "Shared Transformer":
                axes[0, c].fill_between(g["lag"], g["weight_p10"], g["weight_p90"], color=color, alpha=0.15,
                                        label="Transformer, 10th–90th percentile")
            axes[1, c].plot(g["lag"], g["contribution_share"], color=color, marker=marker, ls=ls, label=name)
        title = "1 quarter ahead" if band == "h1" else f"{band[1:].replace('-', '–')} quarters ahead"
        axes[0, c].set_title(title)
        axes[0, c].axhline(0, color="black", lw=0.8)
        for r in (0, 1):
            axes[r, c].set_xticks(range(1, 9))
            axes[r, c].grid(alpha=0.3)
        axes[1, c].set_xlabel("Input quarter (1 = most recent)")
    axes[0, 0].set_ylabel("Weight: change in forecast per\nunit change in that quarter's growth", fontsize=9)
    axes[1, 0].set_ylabel("Share of |contribution|\n(per seed, then averaged)", fontsize=9)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("What the models do with the same eight quarters of house price growth "
                 "(validation 2000–2009, all forecasts)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiment_long.toml")
    ap.add_argument("--panel", default="data/interim/panel_quarterly.csv")
    ap.add_argument("--cpi", default="data/interim/taiwan_cpi_monthly.csv")
    ap.add_argument("--steps", type=int, default=128, help="integration steps for Integrated Gradients")
    ap.add_argument("--reference", default="results/long_hpi_122/validation/preds_transformer__national.csv",
                    help="saved forecasts the retrained models must reproduce")
    ap.add_argument("--tol", type=float, default=1e-5, help="largest allowed |retrained - saved| forecast")
    ap.add_argument("--no-reference-check", action="store_true",
                    help="skip the check against saved forecasts (exploration only)")
    ap.add_argument("--out-dir", type=Path, default=Path("results/explain/validation"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    if cfg["features"]["use"] != ["hpi_qoq"]:
        ap.error("this script explains the HPI-only main experiment")
    samples = load_samples(cfg, args.panel, args.cpi)
    splits = make_splits(cfg, "validation", last_label_origin(samples, end_series(cfg)))
    if args.no_reference_check:
        reference = None
    elif not Path(args.reference).exists():
        ap.error(f"{args.reference} not found; the retrained models must be checked against the saved forecasts "
                 "(use --no-reference-check only for exploration)")
    else:
        reference = pd.read_csv(args.reference)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    meta, arrays, ar_weights, checks = run(cfg, samples, splits, args.steps, reference, args.tol)
    checks.to_csv(args.out_dir / "checks.csv", index=False)
    provenance = {"config": args.config, "config_sha256": sha256(args.config), "panel_sha256": sha256(args.panel),
                  "cpi_sha256": sha256(args.cpi), "reference_sha256": sha256(args.reference) if reference is not None
                  else "", "ig_steps": args.steps, "seeds": ",".join(map(str, cfg["run"]["seeds"]))}
    np.savez_compressed(
        args.out_dir / "row_attributions.npz", series_id=meta["series_id"].to_numpy().astype(str),
        origin_quarter=meta["origin_quarter"].to_numpy().astype(str), split_id=meta["split_id"].to_numpy().astype(str),
        horizons=np.array(samples.horizons), ar_split_id=np.array(ar_weights["split_id"]),
        ar4_weights=np.stack([w for _, w in ar_weights["ar4"]]).astype(np.float32),
        ar8_weights=np.stack([w for _, w in ar_weights["ar8"]]).astype(np.float32),
        **{f"prov_{k}": np.array(str(v)) for k, v in provenance.items()},
        **{k: v.astype(np.float32) for k, v in arrays.items()})
    table = summarize(meta, arrays, ar_weights, list(samples.horizons))
    table.to_csv(args.out_dir / "lag_weights.csv", index=False)
    plot(table, args.out_dir / "lag_weights.png")

    print(f"\nLargest difference from saved forecasts: {checks['max_abs_diff_vs_saved'].max():.2e}; "
          f"largest IG completeness error: {checks['ig_completeness_error'].max():.2e}")
    show = table[table["horizon"].isin(["h3-6", "h16-20"])].pivot_table(
        index=["horizon", "lag"], columns="model", values="weight_mean").round(3)
    print("\nMean weight per input quarter (lag 1 = most recent):")
    print(show.to_string())


if __name__ == "__main__":
    main()
