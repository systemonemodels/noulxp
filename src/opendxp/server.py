"""The HTTP binding (SPEC.md 11): `opendxp serve`, the reference server.

Any machine serves any package over the standard protocol:

    POST /v1/systemone     a request (SPEC.md 3) with an optional "model"; the answer
    GET  /v1/models        the models this server holds (discovery)
    GET  /healthz          liveness

Standard library only. It listens on 127.0.0.1 unless told otherwise, and
refuses to listen anywhere else without a bearer token.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from opendxp import __version__, spec
from opendxp.serving import ServedModel, ServingError, decide

# How much of an oversized body is read and dropped before the connection closes.
DRAIN_BYTES = 16 * spec.MAX_REQUEST_BYTES


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class DecisionServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        models: list[ServedModel],
        *,
        token: str | None = None,
        cors: list[str] | None = None,
        quiet: bool = False,
    ) -> None:
        super().__init__(address, Handler)
        self.models = models
        self.token = token
        self.cors = set(cors or [])
        self.quiet = quiet


class Handler(BaseHTTPRequestHandler):
    server: DecisionServer
    server_version = f"opendxp/{__version__}"
    protocol_version = "HTTP/1.1"

    # --- replies ---------------------------------------------------------------

    def _send(self, status: int, body: Any, headers: dict[str, str] | None = None) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("OpenDXP-Version", spec.PROTOCOL_VERSION)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self._cors_headers()
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, kind: str, message: str) -> None:
        self._send(status, {"error": {"type": kind, "message": message}})

    def _cors_headers(self) -> None:
        origin = self.headers.get("Origin")
        if origin and ("*" in self.server.cors or origin in self.server.cors):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")

    def _authorised(self) -> bool:
        token = self.server.token
        if not token:
            return True
        given = self.headers.get("Authorization", "")
        return given.startswith("Bearer ") and hmac.compare_digest(
            given[len("Bearer ") :].encode(), token.encode()
        )

    def log_message(self, format: str, *args: Any) -> None:
        if not self.server.quiet:
            sys.stderr.write(f"{self.address_string()} {format % args}\n")

    # --- routes ----------------------------------------------------------------

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._cors_headers()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/healthz":
            self._send(200, {"ok": True, "models": len(self.server.models)})
            return
        if not self._authorised():
            self._error(401, "unauthorized", "a bearer token is required")
            return
        if path == spec.HTTP_MODELS_PATH:
            self._send(200, {"object": "list", "data": [m.describe() for m in self.server.models]})
            return
        self._error(404, "not_found", f"no route {path}")

    def _body(self) -> bytes | None:
        """The request body, or None once an error has been sent for it.

        Every POST is read in full before it is answered: bytes left unread on
        a kept-alive connection would be taken for the next request. A body
        over the limit is read and dropped, up to DRAIN_BYTES, then the
        connection is closed.
        """
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self.close_connection = True
            self._error(411, "invalid_request", "Content-Length is required")
            return None
        if length > spec.MAX_REQUEST_BYTES:
            left = min(length, DRAIN_BYTES)
            while left > 0:
                chunk = self.rfile.read(min(left, 1 << 16))
                if not chunk:
                    break
                left -= len(chunk)
            self.close_connection = True
            self._error(
                413, "request_too_large", f"requests are limited to {spec.MAX_REQUEST_BYTES} bytes"
            )
            return None
        return self.rfile.read(length) if length > 0 else b""

    def do_POST(self) -> None:
        body = self._body()
        if body is None:
            return
        path = self.path.split("?", 1)[0]
        if path != spec.HTTP_DECIDE_PATH:
            self._error(404, "not_found", f"no route {path}")
            return
        if not self._authorised():
            self._error(401, "unauthorized", "a bearer token is required")
            return
        try:
            request = json.loads(body or b"null")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._error(400, "invalid_request", f"the body is not JSON: {exc}")
            return
        started = time.perf_counter()
        try:
            answer = decide(self.server.models, request)
        except ServingError as exc:
            self._error(exc.status, exc.kind, exc.message)
            return
        except Exception as exc:
            sys.stderr.write(f"internal error: {type(exc).__name__}: {exc}\n")
            self._error(500, "internal_error", "the model failed to answer; see the server log")
            return
        answer["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        self._send(200, answer)


def serve(
    models: list[ServedModel],
    *,
    host: str = "127.0.0.1",
    port: int = spec.DEFAULT_PORT,
    token: str | None = None,
    cors: list[str] | None = None,
    allow_open: bool = False,
    quiet: bool = False,
) -> DecisionServer:
    """A server bound and ready; call `serve_forever()` on it."""
    if not is_loopback(host) and not token and not allow_open:
        raise ValueError(
            f"refusing to listen on {host} without a token: pass --token (or set "
            "OPENDXP_TOKEN), or --no-auth if the network in front of it authenticates"
        )
    return DecisionServer((host, port), models, token=token, cors=cors, quiet=quiet)
