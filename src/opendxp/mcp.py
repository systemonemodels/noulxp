"""The MCP binding (SPEC.md 14): `opendxp mcp`, every package as a tool for agents.

An agent that speaks the Model Context Protocol calls a decision model like any
other tool: the tool's input is an OpenDXP request (SPEC.md 3) and its output
the answer. The server speaks MCP over stdio, one JSON-RPC message per line,
in both eras of the protocol: the current revision, where every request carries
its version and capabilities in `_meta` and `server/discover` describes the
server, and the earlier revisions that open with an `initialize` handshake.
Nothing but protocol messages is written to stdout.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import IO, Any

from opendxp import __version__, schemas
from opendxp.errors import RequestError
from opendxp.serving import ServedModel, request_problems

# Revisions served per request (modern) and through `initialize` (legacy), newest first.
MODERN = ("2026-07-28",)
LEGACY = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
# The revision that added structuredContent and outputSchema.
STRUCTURED_SINCE = "2025-06-18"

PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR = (
    -32700,
    -32600,
    -32601,
    -32602,
    -32603,
)
UNSUPPORTED_VERSION = -32022

META_VERSION = "io.modelcontextprotocol/protocolVersion"
META_CAPABILITIES = "io.modelcontextprotocol/clientCapabilities"
META_SERVER = "io.modelcontextprotocol/serverInfo"

SERVER_INFO = {"name": "opendxp", "title": "OpenDXP decision models", "version": __version__}
INSTRUCTIONS = (
    "Each tool asks one decision model typed questions about a state and returns a "
    "calibrated probability for every option, in one pass, without generating text. "
    "Put the text to judge in `state` and named questions in `questions`, each with a "
    "`type` and `instructions`: `choice` takes `criteria` mapping option names to "
    "descriptions (or null); `score` takes `criteria` listing levels from lowest to "
    "highest; `noul` asks whether the instructions hold, with optional `true`/`false` "
    "descriptions. Read the decision and its probabilities, and act on it only when the "
    "probability is high enough for what is at stake."
)


class McpError(Exception):
    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


def tool_name(model_id: str, many: bool) -> str:
    """`decide` when the server holds one model; one tool per model otherwise."""
    if not many:
        return "decide"
    return ("decide_" + re.sub(r"[^A-Za-z0-9_.-]", "_", model_id))[:128]


def structured(version: str) -> bool:
    return version in MODERN or version >= STRUCTURED_SINCE


class McpServer:
    def __init__(self, models: list[ServedModel]) -> None:
        many = len(models) > 1
        self.tools = {tool_name(m.id, many): m for m in models}
        if len(self.tools) != len(models):
            raise ValueError("two models map to the same tool name")
        # The legacy revision an `initialize` agreed on, when a client opened that way.
        self.legacy: str | None = None

    # --- one message ----------------------------------------------------------

    def handle(self, message: Any) -> dict[str, Any] | None:
        """The response to one JSON-RPC message, or None for a notification."""
        if not isinstance(message, dict):
            return _error(None, INVALID_REQUEST, "a message must be a JSON-RPC object")
        if "method" not in message:
            return None  # a response from the client: this server sends no requests
        mid = message.get("id")
        if "id" not in message:
            return None  # a notification (initialized, cancelled, ...): nothing to answer
        if message.get("jsonrpc") != "2.0" or not isinstance(message["method"], str):
            return _error(mid, INVALID_REQUEST, "not a JSON-RPC 2.0 request")
        params = message.get("params") or {}
        if not isinstance(params, dict):
            return _error(mid, INVALID_PARAMS, "params must be an object")
        try:
            result = self.dispatch(message["method"], params)
        except McpError as exc:
            return _error(mid, exc.code, exc.message, exc.data)
        except Exception as exc:
            sys.stderr.write(f"internal error: {type(exc).__name__}: {exc}\n")
            return _error(mid, INTERNAL_ERROR, "internal error; see the server's stderr")
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "initialize":
            return self.initialize(params)
        meta = params.get("_meta") if isinstance(params.get("_meta"), dict) else {}
        requested = meta.get(META_VERSION)
        if requested is not None:
            if requested not in MODERN:
                raise McpError(
                    UNSUPPORTED_VERSION,
                    "Unsupported protocol version",
                    {"supported": list(MODERN), "requested": requested},
                )
            if not isinstance(meta.get(META_CAPABILITIES), dict):
                raise McpError(INVALID_PARAMS, f"_meta must carry {META_CAPABILITIES}")
            version, modern = str(requested), True
        elif method == "ping":
            return {}
        elif self.legacy is not None:
            version, modern = self.legacy, False
        else:
            raise McpError(
                INVALID_PARAMS,
                f"_meta must carry {META_VERSION} (or open a session with initialize)",
            )

        if method == "server/discover" and modern:
            result: dict[str, Any] = {
                "supportedVersions": list(MODERN),
                "capabilities": {"tools": {}},
                "instructions": INSTRUCTIONS,
            }
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": self.list_tools(version)}
        elif method == "tools/call":
            result = self.call(params, version)
        else:
            raise McpError(METHOD_NOT_FOUND, f"Method not found: {method}")
        if modern:
            result = {"resultType": "complete", **result, "_meta": {META_SERVER: SERVER_INFO}}
        return result

    # --- methods --------------------------------------------------------------

    def initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        requested = params.get("protocolVersion")
        self.legacy = requested if requested in LEGACY else LEGACY[0]
        return {
            "protocolVersion": self.legacy,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
            "instructions": INSTRUCTIONS,
        }

    def list_tools(self, version: str) -> list[dict[str, Any]]:
        tools = []
        for name, model in self.tools.items():
            tool: dict[str, Any] = {
                "name": name,
                "title": f"Decide with {model.id}",
                "description": _describe(model),
                "inputSchema": schemas.schema("request"),
                "annotations": {"readOnlyHint": True, "openWorldHint": False},
            }
            if structured(version):
                tool["outputSchema"] = schemas.schema("response")
            tools.append(tool)
        return tools

    def call(self, params: dict[str, Any], version: str) -> dict[str, Any]:
        name = params.get("name")
        model = self.tools.get(name) if isinstance(name, str) else None
        if model is None:
            raise McpError(INVALID_PARAMS, f"Unknown tool: {name}")
        arguments = params.get("arguments") or {}
        problems = request_problems(arguments)
        if problems:
            return _tool_error("The request is not valid: " + "; ".join(problems[:5]))
        try:
            answer = model.answer(arguments.get("state", ""), arguments["questions"])
        except RequestError as exc:
            return _tool_error(f"{model.id} cannot answer this request: {exc}")
        result: dict[str, Any] = {
            "content": [{"type": "text", "text": json.dumps(answer, ensure_ascii=False)}],
            "isError": False,
        }
        if structured(version):
            result["structuredContent"] = answer
        return result


def _describe(model: ServedModel) -> str:
    text = (
        f"Ask {model.id}, an OpenDXP {model.package.profile} decision model, typed questions "
        "about a state: choice (pick one option), score (a level on a scale) or noul (yes "
        "or no). Returns the decision and a calibrated probability for every option."
    )
    limits = model.package.limits
    if limits.get("max_options"):
        text += f" At most {limits['max_options']} options per question."
    if model.conformance and model.conformance.get("compatible"):
        text += " OpenDXP compatible: it reproduces its model's own answers."
    return text


def _tool_error(message: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": message}], "isError": True}


def _error(mid: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": mid, "error": error}


def run(server: McpServer, reader: IO[bytes], writer: IO[bytes]) -> None:
    """Serve newline-delimited JSON-RPC until the input closes."""
    for raw in reader:
        line = raw.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            response: dict[str, Any] | None = _error(None, PARSE_ERROR, "Parse error")
        else:
            if isinstance(message, list):
                response = _error(None, INVALID_REQUEST, "batches are not supported")
            else:
                response = server.handle(message)
        if response is not None:
            writer.write(json.dumps(response, ensure_ascii=False).encode("utf-8") + b"\n")
            writer.flush()


def protect_stdout() -> IO[bytes]:
    """The protocol's own copy of stdout, with everything else sent to stderr.

    Libraries that print (Python's print, llama.cpp and ONNX Runtime writing to
    file descriptor 1) would otherwise corrupt the protocol stream.
    """
    protocol = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return os.fdopen(protocol, "wb", buffering=0)
