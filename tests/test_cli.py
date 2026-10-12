"""Test the commands documented for a clean editable checkout."""

import json
import subprocess
import sys

import pytest


@pytest.mark.parametrize("command", ["doctor", "example"])
def test_documented_cli_commands(command):
    result = subprocess.run(
        [sys.executable, "-m", "pure_pino", command],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    report = json.loads(result.stdout)
    assert report.get("new_optimizer_updates", report.get("optimizer_updates")) == 0
    if command == "doctor":
        assert report["real_data_scientific_acceptance"] is False
        assert len(report["native_source_hashes"]) == 4
    else:
        assert report["target_values_read"] == 0
        assert report["lake_accuracy_evaluated"] is False
