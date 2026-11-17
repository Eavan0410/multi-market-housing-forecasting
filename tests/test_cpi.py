import pandas as pd

from housing_forecast.data.cpi import _read_oecd_style, native_yoy, quarterly_yoy


def _raw(rows):
    return pd.DataFrame(rows, columns=["country", "freq", "period", "transformation", "value", "source"])


def test_quarterly_cpi_stored_as_months_is_kept():
    # Australia-style: one observation per quarter in Feb, May, Aug, Nov
    rows = [("A", "M", f"2001-{m:02d}", "GY", v, "OECD") for m, v in zip((2, 5, 8, 11), (1, 2, 3, 4))]
    q = quarterly_yoy(native_yoy(_raw(rows)))
    assert q["quarter"].tolist() == ["2001Q1", "2001Q2", "2001Q3", "2001Q4"]
    assert q["cpi_yoy"].tolist() == [1, 2, 3, 4]


def test_incomplete_monthly_quarter_is_dropped():
    rows = [("B", "M", f"2001-{m:02d}", "GY", 2.0, "OECD") for m in (1, 2, 3, 4, 5, 6, 12)]
    q = quarterly_yoy(native_yoy(_raw(rows)))
    assert q["quarter"].tolist() == ["2001Q1", "2001Q2"]


def test_index_fills_missing_year_on_year_rates():
    rows = [("C", "M", str(p), "_Z", 100 * 1.1 ** (i // 12), "BIS")
            for i, p in enumerate(pd.period_range("2000-01", "2002-12", freq="M"))]
    rows += [("C", "M", "2001-01", "GY", 9.9, "BIS")]          # published value wins
    n = native_yoy(_raw(rows)).set_index("period")
    assert abs(n.loc[pd.Period("2001-01", "M"), "cpi_yoy"] - 9.9) < 1e-9
    assert abs(n.loc[pd.Period("2002-06", "M"), "cpi_yoy"] - 10.0) < 1e-9
    assert n["from_index"].sum() == len(n) - 1


def test_secondary_mirror_rows_are_dropped(tmp_path):
    f = tmp_path / "cpi.csv"
    pd.DataFrame({"Country": ["Korea", "Chile"], "Period": ["2001-01", "2021-01"], "Value": ["1", "2"],
                  "TRANSFORMATION": ["GY", "GY"],
                  "STRUCTURE_ID": ["OECD.SDD.TPS:DSD_PRICES@DF_PRICES_ALL(1.0)", "BIS WS_LONG_CPI 771 (secondary repo: x)"]}
                 ).to_csv(f, index=False)
    d = _read_oecd_style(f, "OECD")
    assert d["country"].tolist() == ["South Korea"]


def test_attach_cpi_yoy_replaces_values_and_leaves_taiwan_for_its_index():
    from housing_forecast.data.cpi import attach_cpi_yoy
    panel = pd.DataFrame({"series_id": ["Japan", "Japan", "Taiwan_national"], "country": ["Japan", "Japan", "Taiwan"],
                          "quarter": ["2001Q1", "2001Q2", "2001Q1"], "hpi": [1.0, 1.0, 1.0], "cpi_yoy": [9.0, 9.0, 9.0]})
    q = pd.DataFrame({"country": ["Japan", "Taiwan"], "quarter": ["2001Q1", "2001Q1"], "cpi_yoy": [1.5, 2.5]})
    out = attach_cpi_yoy(panel, q)
    assert out["cpi_yoy"].iloc[0] == 1.5
    assert out["cpi_yoy"].iloc[1:].isna().all()          # no value for 2001Q2; Taiwan filled later from its index


def test_arx_with_hpi_only_equals_ar():
    import numpy as np
    from housing_forecast.backtesting.infoset import SampleSet
    from housing_forecast.models.baselines import ARPooled, ARXPooled
    rng = np.random.default_rng(0)
    n = 60
    meta = pd.DataFrame({"series_id": ["A"] * n, "country": "A", "geography_level": "national",
                         "origin": pd.period_range("1990Q1", periods=n, freq="Q"),
                         "as_of_date": pd.Timestamp("2000-01-01")})
    s = SampleSet(X=rng.normal(size=(n, 8, 1)), y=rng.normal(size=(n, 2)), y_avail=np.full((n, 2), np.datetime64("NaT")),
                  meta=meta, feature_names=["hpi_qoq"], horizons=(1, 2))
    a, b = ARPooled(), ARXPooled()
    a.fit(s, 0), b.fit(s, 0)
    assert np.allclose(a.predict(s), b.predict(s))
    two = s.replace(X=np.concatenate([s.X, rng.normal(size=(n, 8, 1))], axis=2), feature_names=["hpi_qoq", "cpi_yoy"])
    c = ARXPooled()
    c.fit(two, 0)
    assert c.coefs_["__pooled__"].shape[0] == 1 + 2 * 4      # intercept + 4 lags of each feature
