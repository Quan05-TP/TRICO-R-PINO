"""Restore or update the lossless 600-lake dataset using only the standard library."""

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "600湖泊数据"
MANIFEST = DATA / "manifest.json"


def transfer(source, destination):
    """Copy bytes while computing their digest and length."""
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
        destination.write(block)
        digest.update(block)
        size += len(block)
    return size, digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["unpack", "pack"])
    parser.add_argument(
        "directory", type=Path, help="Restored CSV directory outside this repository"
    )
    parser.add_argument(
        "--file", action="append", help="Original relative path; repeat to select files"
    )
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = manifest["files"]
    selected = {path.replace("\\", "/") for path in args.file or []}
    unknown = selected - {entry["path"] for entry in entries}
    if unknown:
        parser.error(f"Paths absent from manifest: {sorted(unknown)}")
    directory = args.directory.resolve()
    if directory.is_relative_to(DATA.parent):
        parser.error("Choose a restored directory outside the repository")
    for entry in entries:
        if selected and entry["path"] not in selected:
            continue
        restored = directory / entry["path"]
        stored = DATA / entry["stored_path"]
        compressed = entry["encoding"] == "gzip"
        if args.action == "unpack":
            if restored.exists():
                parser.error(f"File already exists: {restored}")
            restored.parent.mkdir(parents=True, exist_ok=True)
            temporary = restored.with_name(restored.name + ".partial")
            opener = gzip.open if compressed else open
            try:
                with opener(stored, "rb") as source, temporary.open("wb") as destination:
                    size, digest = transfer(source, destination)
                if (size, digest) != (entry["bytes"], entry["sha256"]):
                    raise ValueError(f"Size or SHA256 mismatch: {entry['path']}")
                temporary.replace(restored)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            temporary = stored.with_name(stored.name + ".partial")
            try:
                with restored.open("rb") as source, temporary.open("wb") as destination:
                    if compressed:
                        with gzip.GzipFile(
                            filename="", mode="wb", fileobj=destination, compresslevel=6, mtime=0
                        ) as archive:
                            size, digest = transfer(source, archive)
                    else:
                        size, digest = transfer(source, destination)
                temporary.replace(stored)
                entry.update(bytes=size, sha256=digest, stored_bytes=stored.stat().st_size)
                manifest["original_bytes"] = sum(row["bytes"] for row in entries)
                manifest["stored_bytes"] = sum(row["stored_bytes"] for row in entries)
                MANIFEST.write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
            finally:
                temporary.unlink(missing_ok=True)
        print(f"{args.action}: {entry['path']}")
    if args.action == "pack":
        subprocess.run(
            [sys.executable, str(Path(__file__).with_name("check_lake600.py"))], check=True
        )


if __name__ == "__main__":
    main()
