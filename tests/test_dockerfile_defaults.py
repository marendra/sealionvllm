"""Regression tests for SEA-LION runtime defaults in the Docker image."""

from pathlib import Path


DOCKERFILE = Path(__file__).resolve().parent.parent / "Dockerfile"
MODEL_NAME = "aisingapore/Gemma-SEA-LION-v4.5-E2B-IT"


def test_runtime_defaults_are_declared_after_optional_model_download():
    dockerfile = DOCKERFILE.read_text()
    runtime_marker = "# Runtime defaults for the SEA-LION RunPod deployment."
    runtime_start = dockerfile.index(runtime_marker)
    runtime_end = dockerfile.index("# The root handler.py", runtime_start)
    runtime_block = dockerfile[runtime_start:runtime_end]

    assert 'ARG MODEL_NAME=""' in dockerfile[:runtime_start]
    assert "python3 /src/download_model.py" in dockerfile[:runtime_start]
    assert MODEL_NAME not in dockerfile[:runtime_start]

    expected = {
        "MODEL_NAME": MODEL_NAME,
        "MAX_MODEL_LEN": "8192",
        "GPU_MEMORY_UTILIZATION": "0.90",
        "MAX_NUM_SEQS": "32",
        "MAX_NUM_BATCHED_TOKENS": "8192",
        "MAX_CONCURRENCY": "32",
        "DTYPE": "bfloat16",
        "ENABLE_PREFIX_CACHING": "true",
        "ENFORCE_EAGER": "true",
        "TRUST_REMOTE_CODE": "true",
        "VLLM_STARTUP_TIMEOUT": "1200",
        "REQUEST_TIMEOUT": "3600",
        "VLLM_EXTRA_ARGS": "--language-model-only",
    }
    for name, value in expected.items():
        assert f'{name}="{value}"' in runtime_block

    assert "HF_TOKEN" not in runtime_block
