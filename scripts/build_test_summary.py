"""Summarize saved validation/test intervals and block-length checks.

Rescore common-market forecasts without retraining; write results/test_summary.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from housing_forecast.config import load_config
from housing_forecast.evaluation.markets import paired_ratio_bootstrap

BANDS = ["h3-6", "h16-20"]
MAIN_BASELINES = ["transformer_per_market", "ar", "arima", "ar_pooled", "ar_pooled8", "no_change"]


def experiment(rel: Path) -> tuple[str, str | None]:
    """(section, name in the block-length files) for a file path relative to results/."""
    parts = rel.parts                                   # e.g. market_scale_122/history/test/x.csv
    if parts[0] == "long_hpi_122":
        return "main", "main"
    if parts[0] == "market_scale_122":
        order = parts[1] if len(parts) == 4 else "region"
        return f"training_pool_{order}", f"market_scale_{order}"
    if parts[0] == "market_holdout_122":
        return f"holdout_{parts[1]}", f"holdout_{parts[1]}"
    if parts[0] == "market_finetune_122":
        return ("finetune_selected" if "inner_select" in parts else "finetune_fixed"), None
    raise ValueError(folder)


def robust(blocks: pd.DataFrame, name: str | None, a: str, b: str, band: str) -> object:
    """True/False if the interval excludes 1 at every block length; NA if not checked."""
    if name is None:
        return pd.NA
    rows = blocks[blocks["experiment"].eq(name) & blocks["model_a"].eq(a)
                  & blocks["model_b"].eq(b) & blocks["horizon"].eq(band)]
    if rows["block_length"].nunique() < 4:
        return pd.NA
    return bool(((rows["ci_high"] < 1) | (rows["ci_low"] > 1)).all())


def rows_from(path: Path, res: Path, phase: str, blocks: pd.DataFrame) -> list[dict]:
    section, name = experiment(path.relative_to(res))
    ci = pd.read_csv(path)
    out = []
    for r in ci[ci["horizon"].isin(BANDS)].itertuples():
        out.append({"section": section, "phase": phase, "markets": "all", "model_a": r.model_a,
                    "model_b": r.model_b, "horizon": r.horizon, "n_markets": r.n_markets,
                    "median_ratio": r.median_ratio, "ci_low": r.ci_low, "ci_high": r.ci_high,
                    "excludes_1": bool(r.ci_high < 1 or r.ci_low > 1),
                    "excludes_1_all_blocks": robust(blocks, name, r.model_a, r.model_b, r.horizon),
                    "source": str(path.relative_to(res))})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment_long.toml")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/test_summary.csv")
    args = ap.parse_args()
    res = Path(args.results)
    cfg = load_config(args.config)
    blocks = {"validation": pd.read_csv(res / "block_length_sensitivity_122.csv"),
              "test": pd.read_csv(res / "block_length_sensitivity_122_test.csv")}

    rows = []
    for test_file in sorted(res.glob("*_122/**/test/ratio_ci_*.csv")):
        rows += rows_from(test_file, res, "test", blocks["test"])
        val_file = Path(str(test_file).replace("/test/", "/validation/"))
        if val_file.exists():
            rows += rows_from(val_file, res, "validation", blocks["validation"])

    # Main comparison on the markets with forecasts in both periods.
    main_dir = res / "long_hpi_122"
    common = set(pd.read_csv(main_dir / "validation/preds_transformer__national.csv",
                             usecols=["series_id"])["series_id"])
    shared = pd.read_csv(main_dir / "test/preds_transformer__national.csv")
    for base in MAIN_BASELINES:
        preds = pd.concat([shared, pd.read_csv(main_dir / f"test/preds_{base}__national.csv")])
        preds = preds[preds["series_id"].isin(common)]
        ci = paired_ratio_bootstrap(preds, "transformer__national", f"{base}__national", cfg, block_length=20)
        for r in ci[ci["horizon"].isin(BANDS)].itertuples():
            rows.append({"section": "main", "phase": "test", "markets": f"common_{len(common)}",
                         "model_a": "transformer__national", "model_b": f"{base}__national",
                         "horizon": r.horizon, "n_markets": r.n_markets, "median_ratio": r.median_ratio,
                         "ci_low": r.ci_low, "ci_high": r.ci_high,
                         "excludes_1": bool(r.ci_high < 1 or r.ci_low > 1),
                         "excludes_1_all_blocks": pd.NA, "source": "recomputed from test predictions"})

    out = pd.DataFrame(rows)
    order = {"validation": 0, "test": 1}
    out = out.sort_values(["section", "model_b", "model_a", "horizon", "markets", "phase"],
                          key=lambda s: s.map(order) if s.name == "phase" else s, ignore_index=True)
    out.to_csv(args.out, index=False)
    print(f"{len(out)} rows -> {args.out}")
    view = out.assign(ci=out.apply(lambda r: f"{r.median_ratio:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]", axis=1))
    print(view.pivot_table(index=["section", "model_a", "model_b", "markets"], columns=["horizon", "phase"],
                           values="ci", aggfunc="first").to_string())


if __name__ == "__main__":
    main()
