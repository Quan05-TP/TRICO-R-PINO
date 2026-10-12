"""CPU contract controls; no observations or optimizer calls."""

import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic_cpu_fixture():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(13)
    yield
    torch.set_num_threads(previous_threads)
