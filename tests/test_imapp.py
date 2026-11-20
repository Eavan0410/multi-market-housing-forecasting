import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from housing_forecast.backtesting.infoset import build_samples, features_from_config
from housing_forecast.data.imapp import FEATURE, INSTRUMENTS, attach_policy, quarterly_actions


def monthly_rows():
    return pd.DataFrame({
        "source_country": ["TaiwanProvinceofChina"] * 3,
        "iso3": ["TWN"] * 3, "Year": [2001] * 3, "Month": [1, 2, 3],
        "LTV": [1, 0, 0], "DSTI": [0, -1, 0], "LoanR_HH": [0, 0, 1],
    })


def test_quarterly_flow_sums_three_instruments_without_cancelling_missing_data():
    result = quarterly_actions(monthly_rows())
    assert result.loc[0, "quarter"] == "2001Q1"
    assert result.loc[0, FEATURE] == 1
    assert result.loc[0, "n_complete_months"] == 3
    zeros = monthly_rows()
    zeros[INSTRUMENTS] = 0
    assert quarterly_actions(zeros).loc[0, FEATURE] == 0


@pytest.mark.parametrize("missing", ["month", "value"])
def test_incomplete_quarter_stays_missing(missing):
    frame = monthly_rows()
    if missing == "month":
        frame = frame.iloc[:2]
    else:
        frame["DSTI"] = frame["DSTI"].astype(object)
        frame.loc[1, "DSTI"] = "NA"
    assert pd.isna(quarterly_actions(frame).loc[0, FEATURE])


def test_duplicate_month_is_rejected():
    frame = monthly_rows()
    with pytest.raises(ValueError, match="Duplicate"):
        quarterly_actions(pd.concat([frame, frame.iloc[:1]]))


def test_country_mapping_and_uncovered_dates():
    q = quarterly_actions(monthly_rows())
    panel = pd.DataFrame({"country": ["Taiwan", "Taiwan", "Unknown"],
                          "quarter": ["2001Q1", "2001Q2", "2001Q1"]})
    merged = attach_policy(panel, q)
    assert merged.loc[0, FEATURE] == 1
    assert merged.loc[1:, FEATURE].isna().all()
    duplicate = pd.concat([q, q])
    with pytest.raises(ValueError, match="Multiple"):
        attach_policy(panel, duplicate)


def test_matched_inputs_and_no_fill_after_policy_coverage_ends(cfg, monkeypatch):
    prepared = pd.DataFrame({
        "series_id": ["market"] * 14, "country": ["market"] * 14,
        "geography_level": ["national"] * 14,
        "quarter": pd.period_range("2000Q1", periods=14, freq="Q").astype(str),
        "hpi_real": np.arange(100., 114.), "hpi_qoq": [0.01] * 14,
        FEATURE: [0.] * 10 + [np.nan] * 4,
    })
    settings = copy.deepcopy(cfg)
    settings["features"].update(use=["hpi_qoq", FEATURE], lookback=8, ffill_limit=0)
    settings["release_lag_days"][FEATURE] = {"default": 0}
    both = build_samples(prepared, settings, features_from_config(settings))
    assert len(both) == 3
    assert str(both.meta.origin.max()) == "2002Q2"
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_cpi_experiment import hpi_only

    hpi = hpi_only(both)
    np.testing.assert_array_equal(hpi.X, both.X[:, :, :1])
    np.testing.assert_array_equal(hpi.y, both.y)
    pd.testing.assert_frame_equal(hpi.meta, both.meta)
