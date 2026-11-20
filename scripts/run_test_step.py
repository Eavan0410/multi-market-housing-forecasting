"""Run one test step and reuse it only when its outputs are complete."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import pandas as pd

from housing_forecast.backtesting.manifest import check_comparable, read_manifest
from housing_forecast.data.contract import check_alignment, validate_predictions


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_step(directory: Path, name: str, files: list[str], command: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / f".{name}.complete.json"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    identity = {"commit": commit, "command": command, "files": files}
    with (directory / f".{name}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if marker.exists():
            try:
                record = json.loads(marker.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                record = {}
            complete = all((directory / file).is_file() for file in files)
            if complete and record.get("identity") == identity:
                hashes = {file: checksum(directory / file) for file in files}
                if hashes == record.get("sha256"):
                    print(f"Reusing completed step: {name}", flush=True)
                    return

        # Keep interrupted outputs outside the prediction glob before retrying.
        existing = [directory / file for file in files if (directory / file).exists()]
        if existing or marker.exists():
            archive = Path(tempfile.mkdtemp(prefix=f".{name}.incomplete-", dir=directory))
            for path in existing:
                path.rename(archive / path.name)
            if marker.exists():
                marker.rename(archive / marker.name)
        subprocess.run(command, check=True)
        missing = [file for file in files if not (directory / file).is_file()]
        if missing:
            raise ValueError(f"{name} did not produce: {missing}")

        predictions, seeds, manifests = [], {}, {}
        for file in files:
            path = directory / file
            if file.startswith("preds_") and file.endswith(".csv"):
                manifest = read_manifest(path)
                if manifest is None or manifest.get("phase") != "test":
                    raise ValueError(f"{path} needs a test run manifest")
                frame = pd.read_csv(path)
                validate_predictions(frame)
                if frame.empty or set(frame["model"]) != {manifest["model"]}:
                    raise ValueError(f"{path} does not match its manifest")
                predictions.append(frame)
                manifests[file] = manifest
                seeds[manifest["model"]] = manifest["seeds"]
            elif file.endswith(".csv"):
                if pd.read_csv(path).empty:
                    raise ValueError(f"{path} is empty")
        check_comparable(manifests)
        check_alignment(pd.concat(predictions, ignore_index=True), expected_seeds=seeds)
        record = {"identity": identity, "sha256": {file: checksum(directory / file) for file in files}}
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        os.replace(temporary, marker)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    run_step(args.directory, args.name, args.files, command)


if __name__ == "__main__":
    main()
