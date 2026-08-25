"""Validated runtime defaults for the RunPod vLLM worker."""

import math
from collections.abc import MutableMapping
from dataclasses import dataclass

from args_builder import FALSE_VALUES, TRUE_VALUES

DEFAULT_MAX_MODEL_LEN = "8192"
DEFAULT_GPU_MEMORY_UTILIZATION = "0.90"
DEFAULT_ENFORCE_EAGER = "true"
DEFAULT_TRUST_REMOTE_CODE = "true"


class ConfigurationError(ValueError):
    """Raised when worker environment configuration is invalid."""


@dataclass(frozen=True)
class RuntimeConfig:
    model_name: str
    max_model_len: int
    gpu_memory_utilization: float
    enforce_eager: bool
    trust_remote_code: bool


def _set_default(env: MutableMapping[str, str], name: str, default: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        value = default
        env[name] = value
    return value


def _parse_boolean(env: MutableMapping[str, str], name: str, default: str) -> bool:
    value = _set_default(env, name, default).lower()
    if value in TRUE_VALUES:
        env[name] = "true"
        return True
    if value in FALSE_VALUES:
        env[name] = "false"
        return False
    accepted = ", ".join(sorted(TRUE_VALUES | FALSE_VALUES))
    raise ConfigurationError(f"{name} must be a boolean ({accepted}); got {value!r}")


def configure_runtime(env: MutableMapping[str, str]) -> RuntimeConfig:
    """Apply cold-start defaults and validate values before vLLM is launched."""
    model_name = env.get("MODEL_NAME", "").strip()
    if not model_name:
        raise ConfigurationError("MODEL_NAME environment variable is required")
    env["MODEL_NAME"] = model_name

    max_model_len_raw = _set_default(env, "MAX_MODEL_LEN", DEFAULT_MAX_MODEL_LEN)
    try:
        max_model_len = int(max_model_len_raw)
    except ValueError as exc:
        raise ConfigurationError(
            f"MAX_MODEL_LEN must be a positive integer; got {max_model_len_raw!r}"
        ) from exc
    if max_model_len <= 0:
        raise ConfigurationError(
            f"MAX_MODEL_LEN must be a positive integer; got {max_model_len_raw!r}"
        )
    env["MAX_MODEL_LEN"] = str(max_model_len)

    gpu_memory_raw = _set_default(
        env, "GPU_MEMORY_UTILIZATION", DEFAULT_GPU_MEMORY_UTILIZATION
    )
    try:
        gpu_memory_utilization = float(gpu_memory_raw)
    except ValueError as exc:
        raise ConfigurationError(
            "GPU_MEMORY_UTILIZATION must be a number greater than 0 and at most 1; "
            f"got {gpu_memory_raw!r}"
        ) from exc
    if not math.isfinite(gpu_memory_utilization) or not 0 < gpu_memory_utilization <= 1:
        raise ConfigurationError(
            "GPU_MEMORY_UTILIZATION must be a number greater than 0 and at most 1; "
            f"got {gpu_memory_raw!r}"
        )

    enforce_eager = _parse_boolean(env, "ENFORCE_EAGER", DEFAULT_ENFORCE_EAGER)
    trust_remote_code = _parse_boolean(
        env, "TRUST_REMOTE_CODE", DEFAULT_TRUST_REMOTE_CODE
    )

    return RuntimeConfig(
        model_name=model_name,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        trust_remote_code=trust_remote_code,
    )
