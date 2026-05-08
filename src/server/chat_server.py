"""Entry point for the RAG demo chat server.

Usage:
    python3 -m src.server.chat_server [--host HOST] [--port PORT]
"""

import argparse
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG demo chat server")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = parser.parse_args()

    uvicorn.run(
        "src.server.app:app",
        host=args.host,
        port=args.port,
        workers=1,  # one process keeps model singletons in a single memory space
    )


if __name__ == "__main__":
    main()
