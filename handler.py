"""RunPod-discoverable Queue worker entrypoint.

The vLLM process lifecycle and ``runpod.serverless.start`` registration remain
in the upstream ``src/main.py`` implementation. This root module exposes the
handler at RunPod's conventional repository location and delegates startup
without adding inference logic.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src import handler as proxy_handler  # noqa: E402

handler = proxy_handler.handler

__all__ = ["handler"]


def main() -> None:
    """Start the unchanged upstream vLLM lifecycle and RunPod job loop."""
    # src/main.py imports ``handler`` by its historical top-level name. Point
    # that name at the same module exported above so process-health state is
    # shared even when deployment tooling imports this root module first.
    sys.modules["handler"] = proxy_handler

    from main import main as start_worker

    start_worker()


if __name__ == "__main__":
    main()
