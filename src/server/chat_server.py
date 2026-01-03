import argparse
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

from src.pipeline.run_pipeline import run_pipeline

STATIC_DIR = Path(__file__).resolve().parents[2] / "static"
INDEX_FILE = STATIC_DIR / "chat.html"


class ChatHandler(BaseHTTPRequestHandler):
    def _set_headers(self, status: int, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self) -> None:
        self._set_headers(HTTPStatus.NO_CONTENT, "text/plain")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path not in ("/", "/chat.html"):
            self._set_headers(HTTPStatus.NOT_FOUND, "text/plain")
            self.wfile.write(b"Not found")
            return

        if not INDEX_FILE.exists():
            self._set_headers(HTTPStatus.INTERNAL_SERVER_ERROR, "text/plain")
            self.wfile.write(b"Missing static/chat.html")
            return

        self._set_headers(HTTPStatus.OK, "text/html; charset=utf-8")
        self.wfile.write(INDEX_FILE.read_bytes())

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/chat":
            self._set_headers(HTTPStatus.NOT_FOUND, "application/json")
            self.wfile.write(json.dumps({"error": "Not found"}).encode("utf-8"))
            return

        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._set_headers(HTTPStatus.BAD_REQUEST, "application/json")
            self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode("utf-8"))
            return

        query = (payload.get("query") or "").strip()
        if not query:
            self._set_headers(HTTPStatus.BAD_REQUEST, "application/json")
            self.wfile.write(json.dumps({"error": "query is required"}).encode("utf-8"))
            return

        api_key = (payload.get("api_key") or "").strip() or None
        abstain_on_invalid = bool(payload.get("abstain_on_invalid", True))

        try:
            result = run_pipeline(
                query=query,
                api_key=api_key,
                abstain_on_invalid=abstain_on_invalid,
            )
        except Exception as exc:  # noqa: BLE001 - keep demo server simple
            self._set_headers(HTTPStatus.INTERNAL_SERVER_ERROR, "application/json")
            self.wfile.write(
                json.dumps({"error": str(exc)}).encode("utf-8")
            )
            return

        self._set_headers(HTTPStatus.OK, "application/json")
        self.wfile.write(json.dumps(result).encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo chat server")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = parser.parse_args()

    httpd = HTTPServer((args.host, args.port), ChatHandler)
    print(f"Serving on http://{args.host}:{args.port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
