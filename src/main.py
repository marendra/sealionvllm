"""Container entrypoint: spawn `vllm serve`, wait for it, start RunPod serverless.

The worker never imports vLLM. We build the CLI from environment variables
(see args_builder.py), launch `vllm serve` on the loopback interface, poll
/health until the server (and model) is ready, and only then start the RunPod
serverless job loop so no job is pulled before the backend can serve it.
"""

import json
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

from args_builder import build_vllm_args
from download_model import LOCAL_MODEL_ARGS_PATH
from runtime_config import ConfigurationError, RuntimeConfig, configure_runtime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

VLLM_HOST = "127.0.0.1"
VLLM_PORT = os.getenv("VLLM_PORT", "8000")
STARTUP_TIMEOUT = int(os.getenv("VLLM_STARTUP_TIMEOUT", "1200"))  # seconds
HEALTH_POLL_INTERVAL = 2  # seconds
WORKER_START_TIME = time.monotonic()

vllm_process: subprocess.Popen | None = None


def apply_local_model_args() -> None:
    """Load args baked into the image by download_model.py (Option 2 builds).

    The baked model path wins over MODEL_NAME env vars, and HF hub access is
    forced offline since the weights are already on disk.
    """
    if not os.path.exists(LOCAL_MODEL_ARGS_PATH):
        return
    try:
        with open(LOCAL_MODEL_ARGS_PATH) as f:
            local_args = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logging.error("Failed to read %s: %s", LOCAL_MODEL_ARGS_PATH, e)
        return

    logging.info("Using baked-in model args: %s", local_args)
    for key, value in local_args.items():
        if value not in (None, ""):
            os.environ[key] = str(value)
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"


def build_vllm_command() -> list[str]:
    """Return the exact vLLM subprocess command for the current environment."""
    return [
        "vllm",
        "serve",
        "--host",
        VLLM_HOST,
        "--port",
        VLLM_PORT,
        *build_vllm_args(),
    ]


def start_vllm() -> subprocess.Popen:
    argv = build_vllm_command()

    logging.info("Starting vLLM: %s", " ".join(argv))
    # Child gets our stdout/stderr so vLLM logs land in the worker logs.
    return subprocess.Popen(argv)


def wait_for_vllm(proc: subprocess.Popen) -> None:
    """Poll GET /health until vLLM is ready; fail fast if it crashes or times out."""
    url = f"http://{VLLM_HOST}:{VLLM_PORT}/health"

    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM serve exited during startup with code {proc.returncode}")
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=10) as resp:
                if resp.status == 200:
                    logging.info("vLLM is healthy")
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(HEALTH_POLL_INTERVAL)

    raise RuntimeError(f"vLLM did not become healthy within {STARTUP_TIMEOUT}s")


def _forward_signal(signum, _frame):
    logging.info("Received signal %s, shutting down vLLM", signum)
    if vllm_process and vllm_process.poll() is None:
        vllm_process.send_signal(signum)
    sys.exit(128 + signum)


def log_configuration(config: RuntimeConfig) -> None:
    """Log effective startup settings without exposing secrets."""
    logging.info("=== vLLM Configuration ===")
    logging.info("Model: %s", config.model_name)
    logging.info("Max model length: %s", config.max_model_len)
    logging.info("GPU memory utilization: %.2f", config.gpu_memory_utilization)
    logging.info("Enforce eager: %s", str(config.enforce_eager).lower())
    logging.info("Trust remote code: %s", str(config.trust_remote_code).lower())
    logging.info("==========================")


def main() -> None:
    global vllm_process

    apply_local_model_args()

    try:
        config = configure_runtime(os.environ)
    except ConfigurationError as exc:
        logging.critical("Configuration error: %s", exc)
        sys.exit(2)

    log_configuration(config)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _forward_signal)

    vllm_initialization_started = time.monotonic()
    vllm_process = start_vllm()
    try:
        wait_for_vllm(vllm_process)
    except RuntimeError as e:
        logging.error("%s", e)
        sys.exit(1)

    vllm_initialization_elapsed = time.monotonic() - vllm_initialization_started
    worker_startup_elapsed = time.monotonic() - WORKER_START_TIME
    logging.info(
        "startup_timing worker_startup_elapsed_seconds=%.2f "
        "vllm_initialization_elapsed_seconds=%.2f "
        "model_load_elapsed_seconds=unavailable (see vLLM model loading log)",
        worker_startup_elapsed,
        vllm_initialization_elapsed,
    )

    # Import here (not at module import time) so the RunPod SDK and handler
    # start only after the backend is confirmed healthy.
    import handler as proxy_handler
    import runpod

    proxy_handler.vllm_process = vllm_process

    max_concurrency = int(os.getenv("MAX_CONCURRENCY", "30"))
    runpod.serverless.start(
        {
            "handler": proxy_handler.handler,
            "concurrency_modifier": lambda _current: max_concurrency,
            "return_aggregate_stream": True,
        }
    )


if __name__ == "__main__":
    main()
