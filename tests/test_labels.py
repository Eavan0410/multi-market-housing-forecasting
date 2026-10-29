"""Acceptance tests for Round 1 (labels). Run:  pytest tests/test_labels.py -v"""

import numpy as np
import pandas as pd

from housing_forecast.backtesting.labels import make_labels


def _toy_panel():
    # HPI doubles every quarter -> log growth over h quarters = h * log(2)
    quarters = ["2010Q1", "2010Q2", "2010Q3", "2010Q4", "2011Q1", "2011Q2"]
    values = [1, 2, 4, 8, 16, 32]
    return pd.DataFrame({"series_id": "A", "quarter": quarters, "hpi_real": values})


def test_values():
    labels = make_labels(_toy_panel())
    expected = labels["horizon"] * np.log(2)
    assert np.allclose(labels["y_true"], expected)


def test_target_quarter_crosses_year():
    labels = make_labels(_toy_panel())
    row = labels[(labels.origin_quarter == "2010Q4") & (labels.horizon == 2)]
    assert row["target_quarter"].item() == "2011Q2"


def test_row_order_does_not_matter():
    # The Spring 2026 bug: shuffled quarters gave misaligned labels.
    panel = _toy_panel()
    shuffled = panel.sample(frac=1, random_state=0)
    pd.testing.assert_frame_equal(make_labels(panel), make_labels(shuffled))


def test_missing_quarter_gives_no_label():
    # Drop 2010Q3: origin 2010Q2 must have NO h=1 label (not a wrong one).
    panel = _toy_panel()
    panel = panel[panel.quarter != "2010Q3"]
    labels = make_labels(panel)
    hit = labels[(labels.origin_quarter == "2010Q2") & (labels.horizon == 1)]
    assert hit.empty
    # ...but its h=2 label (2010Q4) still exists and is correct.
    h2 = labels[(labels.origin_quarter == "2010Q2") & (labels.horizon == 2)]
    assert np.isclose(h2["y_true"].item(), 2 * np.log(2))


def test_series_do_not_mix():
    a = _toy_panel()
    b = _toy_panel().assign(series_id="B", hpi_real=lambda d: d.hpi_real * 3)
    labels = make_labels(pd.concat([a, b]))
    # Same growth path for both series, so identical y_true per series.
    ya = labels[labels.series_id == "A"]["y_true"].to_numpy()
    yb = labels[labels.series_id == "B"]["y_true"].to_numpy()
    assert np.allclose(ya, yb)
