"""Unit tests for fail-fast worker runtime configuration."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from runtime_config import ConfigurationError, configure_runtime  # noqa: E402


def test_missing_model_name_fails_immediately():
    with pytest.raises(
        ConfigurationError, match="MODEL_NAME environment variable is required"
    ):
        configure_runtime({})


def test_cold_start_defaults_are_applied():
    env = {"MODEL_NAME": "aisingapore/Gemma-SEA-LION-v4.5-E2B-IT"}

    config = configure_runtime(env)

    assert config.model_name == "aisingapore/Gemma-SEA-LION-v4.5-E2B-IT"
    assert config.max_model_len == 8192
    assert config.gpu_memory_utilization == 0.90
    assert config.enforce_eager is True
    assert config.trust_remote_code is True
    assert env == {
        "MODEL_NAME": "aisingapore/Gemma-SEA-LION-v4.5-E2B-IT",
        "MAX_MODEL_LEN": "8192",
        "GPU_MEMORY_UTILIZATION": "0.90",
        "ENFORCE_EAGER": "true",
        "TRUST_REMOTE_CODE": "true",
    }


@pytest.mark.parametrize("value", ["0", "-1", "auto", "8192.5"])
def test_invalid_max_model_len_fails(value):
    with pytest.raises(ConfigurationError, match="MAX_MODEL_LEN"):
        configure_runtime({"MODEL_NAME": "org/model", "MAX_MODEL_LEN": value})


@pytest.mark.parametrize("value", ["0", "-0.1", "1.01", "nan", "many"])
def test_invalid_gpu_memory_utilization_fails(value):
    with pytest.raises(ConfigurationError, match="GPU_MEMORY_UTILIZATION"):
        configure_runtime(
            {
                "MODEL_NAME": "org/model",
                "GPU_MEMORY_UTILIZATION": value,
            }
        )


def test_explicit_false_boolean_is_preserved():
    env = {
        "MODEL_NAME": "org/model",
        "ENFORCE_EAGER": "false",
        "TRUST_REMOTE_CODE": "false",
    }

    config = configure_runtime(env)

    assert config.enforce_eager is False
    assert config.trust_remote_code is False
    assert env["ENFORCE_EAGER"] == "false"
    assert env["TRUST_REMOTE_CODE"] == "false"
