import numpy as np
import pytest

from slm.data import write_records
from slm.model import ModelConfig


@pytest.fixture
def tiny_model_cfg() -> dict:
    return {
        "vocab_size": 50_261,
        "d_model": 32,
        "n_layer": 2,
        "n_head": 4,
        "n_kv_head": 2,
        "ffn_hidden": 64,
        "max_seq_len": 16,
    }


@pytest.fixture
def tiny_model_config(tiny_model_cfg) -> ModelConfig:
    return ModelConfig(**tiny_model_cfg)


@pytest.fixture
def tiny_dataset(tmp_path):
    """Sequencias repetitivas e faceis de aprender, usadas para treinar em segundos."""
    rng = np.random.default_rng(0)
    starts = rng.integers(0, 50, size=(64, 1))
    records = (starts + np.arange(17)) % 50
    directory = tmp_path / "data"
    write_records(directory, records, split_digits=False)
    return directory
