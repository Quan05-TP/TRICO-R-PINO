import gzip
import hashlib

import pytest

from tools.check_lake600 import inspect_file


@pytest.mark.parametrize("compressed", [False, True])
def test_coordinate_and_missing_observation_check(tmp_path, compressed):
    content = (
        "lake_name,Example\nlake_ID,example\nlatitude,30\nlongtitude,100\n\n"
        "datetime_day,datetime_hour/minute,实测温深数据,0m/℃,深度未确定/℃,ERA数据\n"
        "2020-02-29,12:00,,0,,\n2020-02-30,12:00,,,4,\n2020-03-01,,,,,\n"
    ).encode("utf-8-sig")
    path = tmp_path / "lake.csv"
    path.write_bytes(gzip.compress(content) if compressed else content)
    result = inspect_file(path, compressed)
    assert result["sha256"] == hashlib.sha256(content).hexdigest()
    assert result["bytes"] == len(content)
    assert result["rows"] == 3
    assert result["temperature_values"] == 2
    assert result["invalid_date_rows"] == 1
    assert result["empty_temperature_rows"] == 1
    assert result["unknown_depth_values"] == 1
    missing = content.replace(b"latitude,30", b"latitude,")
    path.write_bytes(gzip.compress(missing) if compressed else missing)
    with pytest.raises(ValueError, match="latitude"):
        inspect_file(path, compressed)
