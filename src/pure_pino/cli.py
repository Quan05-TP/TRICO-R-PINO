"""Small local CLI; no automatic training or remote data access."""

import argparse
import importlib.metadata
import json
import runpy
import sys
from pathlib import Path

import torch

from .api import create_model, native_adapter


def doctor() -> dict:
    """Check local imports, current clock and the actual native bindings."""
    model = create_model(width=8, blocks=1, heads=2, spectral_modes=2)
    adapter = native_adapter()
    adapter.bindings.reverify_sources()
    return {
        "status": "LOCAL_IMPORT_AND_NATIVE_SOURCE_HASH_CHECKS_PASS",
        "version": importlib.metadata.version("trico-pure-pino"),
        "torch": torch.__version__,
        "clock_contract": model.cfg.clock_feature_contract_id,
        "native_source_hashes": adapter.bindings.hashes,
        "new_optimizer_updates": 0,
        "real_data_scientific_acceptance": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("doctor", "example"))
    args = parser.parse_args()
    if args.command == "doctor":
        print(json.dumps(doctor(), indent=2))
    else:
        path = Path(__file__).resolve().parents[2] / "examples" / "external_field.py"
        if not path.is_file():
            parser.error("The example requires a repository checkout; run it from the cloned repo.")
        previous_argv = sys.argv
        try:
            sys.argv = [str(path)]
            runpy.run_path(str(path), run_name="__main__")
        finally:
            sys.argv = previous_argv


if __name__ == "__main__":
    main()
