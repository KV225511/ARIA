from __future__ import annotations

import argparse
import asyncio
import sys

import uvicorn


if sys.platform == "win32":
    # Psycopg async connections do not support Windows' Proactor event loop.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ARIA API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    config = uvicorn.Config("app:app", host=args.host, port=args.port)
    server = uvicorn.Server(config)
    loop_factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    with asyncio.Runner(loop_factory=loop_factory) as runner:
        runner.run(server.serve())


if __name__ == "__main__":
    main()
