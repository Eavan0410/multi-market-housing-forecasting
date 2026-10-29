import numpy as np
import pandas as pd
import pytest

from housing_forecast.data.contract import ContractError, check_alignment, validate_predictions

from housing_forecast.evaluation.bootstrap import circular_block_indices, metric_ci, paired_bootstrap
from housing_forecast.evaluation.metrics import direction_accuracy, metrics_table


def _preds(n_origins=40, seed=0):
    rng = np.random.default_rng(seed)
    origins = pd.period_range("2010Q1", periods=n_origins, freq="Q")
    rows = []
    for h in range(1, 5):
        y = rng.normal(0.01 * h, 0.02, n_origins)
        for model, noise in [("good", 0.005), ("bad", 0.05)]:
            rows.append(pd.DataFrame({
                "model": model, "series_id": "Taiwan_national",
                "origin_quarter": origins.astype(str), "target_quarter": (origins + h).astype(str),
                "horizon": h, "y_true": y, "y_pred": y + rng.normal(0, noise, n_origins),
                "seed": 0, "split_id": "val",
                "as_of_date": (origins.end_time.normalize() + pd.Timedelta(days=60)).strftime("%Y-%m-%d"),
                "target_type": "cum_log_growth_real",
            }))
    return pd.concat(rows, ignore_index=True)


def test_direction_accuracy():
    assert direction_accuracy(np.array([1, -1, 1, 0.0]), np.array([2, -3, -1, 5.0])) == 2 / 3


def test_metrics_table_shape():
    t = metrics_table(_preds())
    assert set(t["model"]) == {"good", "bad"} and set(t["horizon"]) == {1, 2, 3, 4}
    assert (t.set_index(["model", "horizon"]).loc["good", "rmse"] < t.set_index(["model", "horizon"]).loc["bad", "rmse"]).all()


def test_block_indices_keep_neighbours():
    idx = circular_block_indices(10, 4, 50, np.random.default_rng(0))
    assert idx.shape == (50, 10)
    assert ((idx[:, 1] - idx[:, 0]) % 10 == 1).all()       # first block is 4 consecutive origins


def test_paired_bootstrap_detects_better_model(cfg):
    cfg["bootstrap"]["n_boot"] = 300
    out = paired_bootstrap(_preds(), "good", "bad", cfg)
    allh = out[out["horizon"] == "all"].iloc[0]
    assert allh["diff"] < 0 and allh["ci_high"] < 0


def test_metric_ci_contains_point(cfg):
    cfg["bootstrap"]["n_boot"] = 300
    out = metric_ci(_preds(), "good", cfg)
    assert ((out["ci_low"] <= out["value"]) & (out["value"] <= out["ci_high"])).all()


def _opposite_seed_predictions():
    base = _preds(n_origins=8)
    stochastic = base[base.model == "good"].copy()
    first = stochastic.assign(seed=0, y_pred=stochastic.y_true + 1)
    second = stochastic.assign(seed=1, y_pred=stochastic.y_true - 1)
    reference = base[base.model == "bad"].copy()
    reference["y_pred"] = reference.y_true
    return pd.concat([first, second, reference], ignore_index=True)


@pytest.mark.parametrize("metric", ["rmse", "mae", "dir_acc"])
def test_bootstrap_uses_mean_seed_metrics(cfg, metric):
    cfg["bootstrap"]["n_boot"] = 50
    preds = _opposite_seed_predictions().sample(frac=1, random_state=42)
    table = metrics_table(preds).set_index(["model", "horizon"])
    paired = paired_bootstrap(preds, "good", "bad", cfg, metric=metric)
    single = metric_ci(preds, "good", cfg, metric=metric).set_index("horizon")
    for h in range(1, 5):
        expected = table.loc[("good", h), metric] - table.loc[("bad", h), metric]
        assert paired.set_index("horizon").loc[h, "diff"] == pytest.approx(expected)
        assert single.loc[h, "value"] == pytest.approx(table.loc[("good", h), metric])
    if metric in ("rmse", "mae"):
        assert np.allclose(single["value"], 1)  # averaging predictions would incorrectly give zero
        assert np.allclose(single[["ci_low", "ci_high"]], 1)


def test_alignment_rejects_missing_row_in_one_seed():
    preds = _opposite_seed_predictions().drop(index=0)
    with pytest.raises(ContractError, match="not aligned"):
        check_alignment(preds)


@pytest.mark.parametrize("field,value", [
    ("y_true", 99.0), ("as_of_date", "2030-01-01"),
    ("target_type", "nominal_growth"), ("split_id", "other_fold"),
    ("target_quarter", "2030Q1"),
])
def test_alignment_rejects_inconsistent_definitions(field, value):
    preds = _opposite_seed_predictions()
    preds.loc[0, field] = value
    with pytest.raises(ContractError):
        check_alignment(preds)


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_predictions_reject_nonfinite(value):
    preds = _preds()
    preds.loc[0, "y_pred"] = value
    with pytest.raises(ContractError):
        validate_predictions(preds)


def test_alignment_rejects_duplicate_across_folds():
    preds = _preds()
    duplicate = preds.iloc[[0]].assign(split_id="another_fold")
    with pytest.raises(ContractError, match="duplicate"):
        check_alignment(pd.concat([preds, duplicate]))


def test_expected_seeds_allow_deterministic_reference_but_catch_missing_seed():
    preds = _opposite_seed_predictions()
    expected = {"good": [0, 1], "bad": [0]}
    check_alignment(preds, expected_seeds=expected)
    with pytest.raises(ContractError, match="seeds"):
        check_alignment(preds[~((preds.model == "good") & (preds.seed == 1))], expected_seeds=expected)


def test_bootstrap_checks_inputs_before_scoring(cfg):
    preds = _opposite_seed_predictions()
    preds.loc[0, "y_true"] = 100
    with pytest.raises(ContractError, match="y_true"):
        paired_bootstrap(preds, "good", "bad", cfg)
    with pytest.raises(ContractError, match="y_true"):
        metric_ci(preds, "good", cfg)
