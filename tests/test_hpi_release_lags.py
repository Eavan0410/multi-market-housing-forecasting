import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("availability", ROOT / "scripts/build_availability_matrix.py")
availability = importlib.util.module_from_spec(spec)
spec.loader.exec_module(availability)


@pytest.fixture
def lag_table():
    return pd.read_csv(ROOT / "docs/hpi_release_lags.csv", dtype=str, keep_default_na=False)


def test_matrix_uses_all_evaluation_markets_and_raw_coverage():
    matrix = availability.build_matrix(ROOT / "data/raw")
    hpi = matrix.loc[matrix["variable"].eq("hpi")].set_index("country")
    listed = pd.read_csv(ROOT / "docs/market_list.csv")
    listed = listed.loc[listed["validation_origins_2000_2009"] > 0].copy()
    listed["country"] = listed["series_id"].replace({"Taiwan_national": "Taiwan"})
    assert len(matrix) == 73
    assert set(hpi.index) == set(listed["country"])
    for row in listed.itertuples():
        assert hpi.loc[row.country, "first_period"] == row.hpi_start
        assert hpi.loc[row.country, "last_period"] == row.hpi_end
        assert hpi.loc[row.country, "valid_observations"] == row.n_quarters
    assert hpi["common_valid_observations"].nunique() == 1
    assert set(matrix.loc[matrix["variable"].ne("hpi"), "country"]) == set(availability.COUNTRIES.values())


def test_dated_evidence_updates_the_matrix_without_changing_configs(lag_table, tmp_path):
    config = ROOT / "configs/experiment_long.toml"
    before = config.read_bytes()
    selected = lag_table["series_id"].eq("Germany")
    lag_table.loc[selected, ["lag_days", "reference_period", "release_date", "evidence_url", "status"]] = [
        "120", "2024Q4", "2025-04-30", "https://example.org/test-release", "release_example"]
    path = tmp_path / "lags.csv"
    lag_table.to_csv(path, index=False)
    matrix = availability.build_matrix(ROOT / "data/raw", release_lags=path)
    row = matrix.loc[matrix["country"].eq("Germany")].iloc[0]
    assert row["publication_lag_status"] == "Release example"
    assert "120 calendar days" in row["publication_lag"]
    assert "does not establish a fixed historical release lag" in row["publication_lag"]
    assert config.read_bytes() == before


@pytest.mark.parametrize("field,value", [("lag_days", "100"), ("country", "Japan"), ("status", "done")])
def test_incomplete_or_misaligned_evidence_is_rejected(lag_table, tmp_path, field, value):
    lag_table.loc[lag_table["series_id"].eq("Germany"), field] = value
    path = tmp_path / "lags.csv"
    lag_table.to_csv(path, index=False)
    with pytest.raises(ValueError):
        availability.hpi_release_evidence(path, lag_table["series_id"])


def test_duplicate_market_is_rejected(lag_table, tmp_path):
    path = tmp_path / "lags.csv"
    pd.concat([lag_table, lag_table.iloc[:1]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="exactly once"):
        availability.hpi_release_evidence(path, lag_table["series_id"])


def test_mismatched_release_date_is_rejected(lag_table, tmp_path):
    selected = lag_table["series_id"].eq("Germany")
    lag_table.loc[selected, ["lag_days", "reference_period", "release_date", "evidence_url", "status"]] = [
        "30", "2024Q4", "2025-04-30", "https://example.org/test-release", "release_example"]
    path = tmp_path / "lags.csv"
    lag_table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="does not match"):
        availability.hpi_release_evidence(path, lag_table["series_id"])


def test_lags_longer_than_config_are_reported(lag_table, tmp_path):
    for sid, values in {
        "Germany": ["120", "2024Q4", "2025-04-30", "https://example.org/de", "release_example"],
        "France": ["60", "2024Q4", "2025-03-01", "https://example.org/fr", "release_example"],
    }.items():
        lag_table.loc[lag_table["series_id"].eq(sid),
                      ["lag_days", "reference_period", "release_date", "evidence_url", "status"]] = values
    path = tmp_path / "lags.csv"
    lag_table.to_csv(path, index=False)
    found = availability.lags_above_config(path)
    assert {(sid, lag, assumed) for _, sid, lag, assumed, _ in found} == {("Germany", 120, 100)}
    assert {config for config, *_ in found} == {"experiment.toml", "experiment_long.toml"}
    assert availability.lags_above_config(ROOT / "docs/hpi_release_lags.csv") == []
