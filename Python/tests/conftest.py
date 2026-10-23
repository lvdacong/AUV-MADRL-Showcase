from __future__ import annotations

import pytest
import torch

from auv_madrl.config import load_project_config


@pytest.fixture(scope="session")
def project_config():
    return load_project_config()


@pytest.fixture(scope="session")
def device():
    if not torch.cuda.is_available():
        pytest.skip("The formal Python runtime requires CUDA")
    return torch.device("cuda")
