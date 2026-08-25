"""Tests for the root RunPod handler entrypoint."""

import importlib.util
import inspect
from pathlib import Path


def test_root_entrypoint_exposes_queue_handler_and_upstream_main():
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("runpod_worker_entrypoint", root / "handler.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert inspect.isasyncgenfunction(module.handler)
    assert module.handler is module.proxy_handler.handler
    assert callable(module.main)
    assert module.SRC_DIR == root / "src"
