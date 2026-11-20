import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "run_test_step", Path(__file__).resolve().parents[1] / "scripts/run_test_step.py"
)
step = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step)


@pytest.fixture
def output_step(tmp_path):
    writer = tmp_path / "writer.py"
    writer.write_text('''import json, sys
from pathlib import Path
import pandas as pd
out = Path(sys.argv[1])
count = out / "attempts.txt"
count.write_text(str(int(count.read_text()) + 1 if count.exists() else 1))
pd.DataFrame([dict(model="ar", series_id="market", origin_quarter="2015Q1",
    as_of_date="2015-07-31", target_quarter="2015Q2", horizon=1,
    y_true=0.1, y_pred=0.0, target_type="cum_log_growth_real", seed=0,
    split_id="tes_2015Q1")]).to_csv(out / "preds_ar.csv", index=False)
if (out / "interrupt").exists():
    raise SystemExit(1)
(out / "preds_ar.meta.json").write_text(json.dumps(
    dict(phase="test", model="ar", seeds=[0])))
pd.DataFrame([dict(status="fitted")]).to_csv(out / "diagnostics.csv", index=False)
''')
    files = ["preds_ar.csv", "preds_ar.meta.json", "diagnostics.csv"]
    command = [sys.executable, str(writer), str(tmp_path)]
    return tmp_path, files, command


def test_interrupted_step_retries_and_complete_step_is_reused(output_step):
    folder, files, command = output_step
    (folder / "interrupt").touch()
    with pytest.raises(subprocess.CalledProcessError):
        step.run_step(folder, "example", files, command)
    assert not (folder / ".example.complete.json").exists()
    (folder / "interrupt").unlink()
    step.run_step(folder, "example", files, command)
    assert (folder / "attempts.txt").read_text() == "2"
    assert len(list(folder.glob(".example.incomplete-*/preds_ar.csv"))) == 1
    step.run_step(folder, "example", files, command)
    assert (folder / "attempts.txt").read_text() == "2"


@pytest.mark.parametrize("damage", ["prediction", "diagnostics", "marker"])
def test_damaged_outputs_trigger_retry(output_step, damage):
    folder, files, command = output_step
    step.run_step(folder, "example", files, command)
    if damage == "prediction":
        (folder / "preds_ar.csv").write_text("truncated\n")
    elif damage == "diagnostics":
        (folder / "diagnostics.csv").unlink()
    else:
        (folder / ".example.complete.json").write_text("{")
    step.run_step(folder, "example", files, command)
    assert (folder / "attempts.txt").read_text() == "2"
    assert json.loads((folder / ".example.complete.json").read_text())["sha256"]


def test_cli_checks_complete_group(output_step):
    folder, files, command = output_step
    subprocess.run([sys.executable, str(Path(step.__file__)), "--directory", str(folder),
                    "--name", "example", "--files", *files, "--", *command], check=True)
    assert (folder / ".example.complete.json").exists()
