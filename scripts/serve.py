#!/usr/bin/env python3
"""Run the CORTEX-0 HTTP API with uvicorn."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--db-path", default="cortex.sqlite")
    parser.add_argument("--workspace", default=str(ROOT))
    parser.add_argument("--scheduler", action="store_true", help="enable the proactive scheduler")
    parser.add_argument("--scheduler-interval", type=float, default=60.0)
    args = parser.parse_args()

    if args.model_dir:
        os.environ["CORTEX_MODEL_DIR"] = args.model_dir
    os.environ["CORTEX_DB_PATH"] = args.db_path
    os.environ["CORTEX_WORKSPACE"] = args.workspace
    if args.scheduler:
        os.environ["CORTEX_ENABLE_SCHEDULER"] = "1"
        os.environ["CORTEX_SCHEDULER_INTERVAL"] = str(args.scheduler_interval)

    import uvicorn

    uvicorn.run("cortex.api.app:app", host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
