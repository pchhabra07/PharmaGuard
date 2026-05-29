"""
PharmaGuard — Phase 5: REST API Runner
==========================================
Top-level script that starts the FastAPI prediction server.

Matches the project convention of ``run_*.py`` entry points at
the project root.

Usage::

    python run_api.py                  # default: localhost:8000
    python run_api.py --port 9000      # custom port
    python run_api.py --host 0.0.0.0   # bind to all interfaces
"""

import argparse
import logging
import sys

import uvicorn
import yaml
from pathlib import Path

# ── Configure logging ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("PharmaGuard.API")

# ── Project root ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    """Parse command-line arguments for host and port overrides."""
    parser = argparse.ArgumentParser(
        description="PharmaGuard — Start the prediction API server",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind the server to (default: from config or 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind the server to (default: from config or 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    return parser.parse_args()


def main():
    """Start the FastAPI server using uvicorn."""
    args = parse_args()

    # Load config for defaults
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    api_cfg = config.get("api", {})
    host = args.host or api_cfg.get("host", "0.0.0.0")
    port = args.port or api_cfg.get("port", 8000)

    logger.info("=" * 60)
    logger.info("PharmaGuard — Phase 5: REST API Server")
    logger.info("=" * 60)
    logger.info("  Host:    %s", host)
    logger.info("  Port:    %d", port)
    logger.info("  Reload:  %s", args.reload)
    logger.info("  Docs:    http://%s:%d/docs", "localhost" if host == "0.0.0.0" else host, port)
    logger.info("=" * 60)

    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
