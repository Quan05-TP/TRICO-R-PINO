import gzip
import hashlib
import json
import sys

import pytest

from tools import lake600


@pytest.mark.parametrize("encoding", ["gzip", "plain"])
def test_restore_edit_and_repack(tmp_path, monkeypatch, encoding):
    checks = []
    monkeypatch.setattr(lake600.subprocess, "run", lambda *a, **k: checks.append((a, k)))
    data = tmp_path / "repo" / "data"
    data.mkdir(parents=True)
    manifest_path = data / "manifest.json"
    original = "date,temperature\n2026-01-01,4.5\n".encode()
    stored = data / ("lake.csv.gz" if encoding == "gzip" else "lake.csv")
    stored.write_bytes(gzip.compress(original) if encoding == "gzip" else original)
    entry = {
        "path": "湖泊/lake.csv",
        "stored_path": stored.name,
        "encoding": encoding,
        "bytes": len(original),
        "stored_bytes": stored.stat().st_size,
        "sha256": hashlib.sha256(original).hexdigest(),
    }
    manifest_path.write_text(json.dumps({"files": [entry]}), encoding="utf-8")
    monkeypatch.setattr(lake600, "DATA", data)
    monkeypatch.setattr(lake600, "MANIFEST", manifest_path)
    working = tmp_path / "working"
    monkeypatch.setattr(sys, "argv", ["lake600", "unpack", str(working)])
    lake600.main()
    restored = working / entry["path"]
    assert restored.read_bytes() == original
    modified = original + b"2026-01-02,5.0\n"
    restored.write_bytes(modified)
    monkeypatch.setattr(sys, "argv", ["lake600", "pack", str(working), "--file", entry["path"]])
    lake600.main()
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(checks) == 1
    assert checks[0][1]["check"] is True
    assert saved["files"][0]["sha256"] == hashlib.sha256(modified).hexdigest()
    assert saved["original_bytes"] == len(modified)
    assert saved["stored_bytes"] == stored.stat().st_size
    second = tmp_path / "second"
    monkeypatch.setattr(sys, "argv", ["lake600", "unpack", str(second)])
    lake600.main()
    assert (second / entry["path"]).read_bytes() == modified
    saved["files"][0]["sha256"] = "incorrect"
    manifest_path.write_text(json.dumps(saved), encoding="utf-8")
    third = tmp_path / "third"
    monkeypatch.setattr(sys, "argv", ["lake600", "unpack", str(third)])
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        lake600.main()
    assert not (third / entry["path"]).exists()
