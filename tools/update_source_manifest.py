"""Record publication hashes without changing the frozen source provenance."""

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate without writing")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = root / "docs" / "SOURCE_PROVENANCE.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mismatches = []
    for entry in manifest["files"]:
        source = (root / entry["publication_path"]).resolve()
        if not source.is_relative_to(root):
            raise ValueError("Publication member leaves repository")
        current = hashlib.sha256(source.read_bytes()).hexdigest()
        frozen = (
            "/_features/" in entry["publication_path"] or "/_native/" in entry["publication_path"]
        )
        if frozen and current != entry["original_sha256"]:
            raise ValueError("Frozen source was changed: " + entry["publication_path"])
        if args.check:
            if entry.get("publication_sha256") != current:
                mismatches.append(entry["publication_path"])
        else:
            entry["publication_sha256"] = current
            entry["publication_byte_exact_to_original"] = current == entry["original_sha256"]
            if not frozen:
                if current != entry["original_sha256"]:
                    entry["modification_type"] = entry["modification_type"].replace(
                        "byte_exact_copy", "Ruff_formatting_and_import_cleanup"
                    )
                entry["publication_style"] = (
                    "Ruff formatting and import cleanup; portable namespace"
                )
    if mismatches:
        raise SystemExit("Publication hashes need review/update: " + ", ".join(mismatches))
    if not args.check:
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "SOURCE_MANIFEST_CHECKED", "members": len(manifest["files"])}))


if __name__ == "__main__":
    main()
