import pytest

from housing_forecast.backtesting.manifest import check_comparable, make_manifest, manifest_path
from housing_forecast.data.contract import ContractError


@pytest.fixture
def files(tmp_path):
    panel, cpi = tmp_path / "panel.csv", tmp_path / "cpi.csv"
    panel.write_text("a\n1\n")
    cpi.write_text("b\n2\n")
    return panel, cpi


def _m(cfg, files, **kw):
    panel, cpi = files
    args = dict(experiment="e", phase="validation", model="ar", base_model="ar", pool="target_only",
                stochastic=False, seeds=[0], panel_path=panel, cpi_path=cpi)
    args.update(kw)
    return make_manifest(cfg, **args)


def test_manifest_path():
    assert manifest_path("results/x/validation/preds_ar.csv").name == "preds_ar.meta.json"


def test_different_pools_are_comparable(cfg, files):
    a = _m(cfg, files)
    b = _m(cfg, files, model="ar__overseas", pool="overseas")
    check_comparable({"a": a, "b": b})


def test_different_features_are_rejected(cfg, files):
    a = _m(cfg, files)
    cfg["features"]["use"] = ["hpi_qoq"]
    b = _m(cfg, files)
    with pytest.raises(ContractError, match="features"):
        check_comparable({"a": a, "b": b})


def test_different_data_file_is_rejected(cfg, files):
    a = _m(cfg, files)
    files[0].write_text("a\n2\n")
    b = _m(cfg, files)
    with pytest.raises(ContractError, match="panel_sha256"):
        check_comparable({"a": a, "b": b})
