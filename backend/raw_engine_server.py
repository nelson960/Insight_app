from __future__ import annotations

import socket
import sys

import uvicorn

from backend.services.raw_engine_server.app import create_app
from backend.services.raw_engine_server.config import EngineConfig

app = create_app()


def _warn_dual_process() -> None:
    print(
        "Warning: Raw Engine runs as a separate process. If Insight desktop is also running,",
        "the model will be loaded twice and memory usage will increase.",
        file=sys.stderr,
    )


def _port_available(host: str, port: int) -> bool:
    try:
        with socket.create_server((host, port)):
            return True
    except OSError:
        return False


def _select_port(host: str, port: int, *, max_tries: int = 20) -> int:
    for candidate in range(port, port + max_tries):
        if _port_available(host, candidate):
            return candidate
    raise RuntimeError(f"No available port found in range {port}-{port + max_tries - 1}")


if __name__ == "__main__":
    config = EngineConfig.from_env()
    _warn_dual_process()
    selected_port = _select_port(config.host, config.port)
    if selected_port != config.port:
        print(
            f"Port {config.port} unavailable, falling back to {selected_port}.",
            file=sys.stderr,
        )
        config.port = selected_port
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")
