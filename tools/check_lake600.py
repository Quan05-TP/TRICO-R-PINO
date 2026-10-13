"""Check every delivered file and report missing lake data after each change."""

import argparse
import csv
import gzip
import hashlib
import json
import math
from datetime import date
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "600湖泊数据"


def inspect_file(path, compressed=False, observation=True):
    digest = hashlib.sha256()
    size = 0
    opener = gzip.open if compressed else open
    with opener(path, "rb") as stream:
        if not observation:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
                size += len(block)
            return {"bytes": size, "sha256": digest.hexdigest()}

        def lines():
            nonlocal size
            for line in stream:
                size += len(line)
                digest.update(line)
                yield line.decode("utf-8-sig")

        reader = csv.reader(lines())
        meta = {}
        for header in reader:
            if header and header[0] == "datetime_day":
                break
            if len(header) >= 2:
                meta[header[0]] = header[1]
        else:
            raise ValueError(f"Missing observation header: {path}")
        for key, limit in [("latitude", 90), ("longtitude", 180)]:
            value = float(meta.get(key) or "nan")
            if not math.isfinite(value) or abs(value) > limit:
                raise ValueError(f"Missing or invalid {key}: {path}")
        if not meta.get("lake_ID") or not meta.get("lake_name"):
            raise ValueError(f"Missing lake identity: {path}")
        end = header.index("ERA数据")
        unknown = header.index("深度未确定/℃") if "深度未确定/℃" in header else None
        metrics = dict(
            rows=0,
            invalid_date_rows=0,
            empty_temperature_rows=0,
            temperature_values=0,
            unknown_depth_values=0,
        )
        dates = {}
        for row in reader:
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(f"Observation column count changed: {path}")
            metrics["rows"] += 1
            count = sum(map(bool, row[3:end]))
            metrics["temperature_values"] += count
            metrics["empty_temperature_rows"] += count == 0
            metrics["unknown_depth_values"] += bool(unknown is not None and row[unknown])
            day = row[0].strip().replace("/", "-")
            if day not in dates:
                try:
                    date.fromisoformat(day)
                    dates[day] = True
                except ValueError:
                    dates[day] = False
            metrics["invalid_date_rows"] += not dates[day]
        if not metrics["rows"] or not metrics["temperature_values"]:
            raise ValueError(f"Empty observation file: {path}")
    return {
        "bytes": size,
        "sha256": digest.hexdigest(),
        "lake_id": meta["lake_ID"],
        "latitude": float(meta["latitude"]),
        "longitude": float(meta["longtitude"]),
        **metrics,
    }


def check_dataset(data=DATA, original_directory=None):
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    baseline = json.loads((data / "missingness_baseline.json").read_text(encoding="utf-8"))
    lakes = {}
    for entry in manifest["files"]:
        observation = "/完整观测/" in entry["path"]
        path = (
            original_directory / entry["path"]
            if original_directory
            else data / entry["stored_path"]
        )
        result = inspect_file(
            path, not original_directory and entry["encoding"] == "gzip", observation
        )
        if (result["bytes"], result["sha256"]) != (entry["bytes"], entry["sha256"]):
            raise ValueError(f"Manifest mismatch: {entry['path']}")
        if observation:
            old = baseline[entry["path"]]
            if result["lake_id"] != old["lake_id"]:
                raise ValueError(f"Lake identity changed: {entry['path']}")
            for key in ["invalid_date_rows", "empty_temperature_rows", "unknown_depth_values"]:
                if result[key] > old[key]:
                    raise ValueError(f"New missing data ({key}): {entry['path']}")
            if result["temperature_values"] < old["temperature_values"]:
                raise ValueError(f"Observation values removed: {entry['path']}")
            lakes[entry["path"]] = result
    if set(lakes) != set(baseline) or len(lakes) != 600:
        raise ValueError("Lake file inventory changed")
    print(
        json.dumps(
            {
                "checked_files": len(manifest["files"]),
                "checked_lakes": len(lakes),
                "missing_coordinates": 0,
                "empty_observation_files": 0,
                "new_missing_data": 0,
                "existing_missing": {
                    k: sum(r[k] for r in lakes.values())
                    for k in ["invalid_date_rows", "empty_temperature_rows", "unknown_depth_values"]
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return lakes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-directory", type=Path)
    args = parser.parse_args()
    check_dataset(original_directory=args.original_directory)
