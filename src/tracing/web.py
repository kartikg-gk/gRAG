"""Serve saved traces and the bundled viewer on the loopback interface."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlsplit

from .schema import Trace, to_dict
from .store import load


def discover(path: Path) -> list[tuple[Path, Trace]]:
    """Read one trace, or every valid trace in a directory, newest first."""
    if path.is_file():
        return [(path, load(path))]
    if not path.is_dir():
        raise ValueError(f"no such trace file or directory: {path}")
    runs = []
    for candidate in sorted(path.glob("*.json"), reverse=True):
        try:
            runs.append((candidate, load(candidate)))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if not runs:
        raise ValueError(f"no readable trace JSON files in: {path}")
    return sorted(runs, key=lambda run: (
        run[1].started_at.timestamp() if run[1].started_at else run[0].stat().st_mtime,
        run[0].name,
    ), reverse=True)


def make_handler(runs: list[tuple[Path, Trace]]) -> type[BaseHTTPRequestHandler]:
    class ViewerHandler(BaseHTTPRequestHandler):
        def _send(self, payload: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            # A public website must not read local traces through DNS rebinding.
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if self.headers.get("Host") not in allowed:
                self._send(b"Forbidden host", "text/plain; charset=utf-8", 403)
                return
            origin = self.headers.get("Origin")
            if origin is not None and origin not in {f"http://{host}" for host in allowed}:
                self._send(b"Forbidden origin", "text/plain; charset=utf-8", 403)
                return
            path = urlsplit(self.path).path
            if path == "/api/local/runs":
                data = [
                    {"id": str(index), "name": name.name, "query": trace.query,
                     "started_at": trace.started_at.isoformat() if trace.started_at else None}
                    for index, (name, trace) in enumerate(runs)
                ]
                self._send(json.dumps(data).encode(), "application/json; charset=utf-8")
                return
            if path.startswith("/api/local/traces/"):
                index_text = path.removeprefix("/api/local/traces/")
                if index_text.isdecimal() and int(index_text) < len(runs):
                    self._send(json.dumps(to_dict(runs[int(index_text)][1])).encode(),
                               "application/json; charset=utf-8")
                else:
                    self._send(b"Not found", "text/plain; charset=utf-8", 404)
                return
            if path == "/":
                path = "/viewer.html"
            parts = path.lstrip("/").split("/")
            if not (
                parts == ["viewer.html"]
                or (len(parts) == 2 and parts[0] == "assets"
                    and re.fullmatch(r"[A-Za-z0-9_.-]+", parts[1])
                    and parts[1] not in (".", ".."))
            ):
                self._send(b"Not found", "text/plain; charset=utf-8", 404)
                return
            asset = files(__package__).joinpath("ui", *parts)
            if not asset.is_file():
                self._send(b"Not found", "text/plain; charset=utf-8", 404)
                return
            body = asset.read_bytes()
            if path == "/viewer.html":
                body = body.replace(b"</head>",
                    b"<script>window.__GRAPHRAG_LOCAL_VIEWER__=true</script></head>")
            content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
            self._send(body, content_type)

    return ViewerHandler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Open a trace or directory in the local viewer")
    parser.add_argument("path", type=Path, help="trace JSON file or directory of traces")
    parser.add_argument("--port", type=int, default=4630, help="loopback port (default: 4630)")
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    try:
        runs = discover(args.path)
        if not files(__package__).joinpath("ui", "viewer.html").is_file():
            raise ValueError("viewer assets are missing from this installation")
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(runs))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"graphrag-view: {exc}", file=sys.stderr)
        return 1
    address = f"http://127.0.0.1:{server.server_port}/"
    print(f"Viewing {len(runs)} trace(s) at {address}", flush=True)
    if not args.no_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
