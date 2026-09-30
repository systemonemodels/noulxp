"""The MCP binding (SPEC.md 12): both protocol eras, on the toy package."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from noulxp import schemas
from noulxp.mcp import (
    INVALID_PARAMS,
    LEGACY,
    METHOD_NOT_FOUND,
    MODERN,
    PARSE_ERROR,
    UNSUPPORTED_VERSION,
    McpServer,
    run,
    tool_name,
)
from noulxp.serving import ServedModel, load_models
from test_conformance import TOY_REQUESTS, toy_package  # noqa: F401 - a fixture

REQUEST = TOY_REQUESTS[0]["request"]
META = {
    "io.modelcontextprotocol/protocolVersion": MODERN[0],
    "io.modelcontextprotocol/clientCapabilities": {},
    "io.modelcontextprotocol/clientInfo": {"name": "test", "version": "1"},
}


@pytest.fixture()
def models(toy_package: Path) -> Iterator[list[ServedModel]]:  # noqa: F811
    loaded = load_models([toy_package], device="cpu", log=lambda *_: None)
    yield loaded
    for model in loaded:
        model.close()


def rpc(server: McpServer, method: str, params: dict[str, Any] | None = None, mid: int = 1):  # type: ignore[no-untyped-def]
    message: dict[str, Any] = {"jsonrpc": "2.0", "id": mid, "method": method}
    if params is not None:
        message["params"] = params
    return server.handle(message)


def test_a_legacy_session(models: list[ServedModel]) -> None:
    server = McpServer(models)
    init = rpc(server, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}})
    assert init["result"]["protocolVersion"] == "2025-06-18"
    assert init["result"]["capabilities"] == {"tools": {"listChanged": False}}
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None

    tools = rpc(server, "tools/list", {})["result"]["tools"]
    assert [t["name"] for t in tools] == ["decide"]
    assert tools[0]["inputSchema"] == schemas.schema("request")
    assert tools[0]["outputSchema"] == schemas.schema("response")
    assert "resultType" not in rpc(server, "tools/list", {})["result"]

    result = rpc(server, "tools/call", {"name": "decide", "arguments": REQUEST})["result"]
    assert result["isError"] is False
    direct = models[0].runtime.predict(REQUEST["state"], REQUEST["questions"])
    assert result["structuredContent"]["answers"] == direct["answers"]
    assert json.loads(result["content"][0]["text"]) == result["structuredContent"]


def test_older_legacy_revisions_get_text_only(models: list[ServedModel]) -> None:
    server = McpServer(models)
    rpc(server, "initialize", {"protocolVersion": "2024-11-05"})
    assert "outputSchema" not in rpc(server, "tools/list", {})["result"]["tools"][0]
    result = rpc(server, "tools/call", {"name": "decide", "arguments": REQUEST})["result"]
    assert "structuredContent" not in result and result["content"][0]["type"] == "text"


def test_an_unknown_legacy_revision_is_answered_with_the_newest(
    models: list[ServedModel],
) -> None:
    server = McpServer(models)
    init = rpc(server, "initialize", {"protocolVersion": "1999-01-01"})
    assert init["result"]["protocolVersion"] == LEGACY[0]


def test_the_modern_era(models: list[ServedModel]) -> None:
    server = McpServer(models)
    found = rpc(server, "server/discover", {"_meta": META})["result"]
    assert found["resultType"] == "complete"
    assert found["supportedVersions"] == list(MODERN)
    assert found["capabilities"] == {"tools": {}}
    assert found["_meta"]["io.modelcontextprotocol/serverInfo"]["name"] == "noulxp"

    listed = rpc(server, "tools/list", {"_meta": META})["result"]
    assert listed["resultType"] == "complete" and listed["tools"][0]["name"] == "decide"

    result = rpc(server, "tools/call", {"_meta": META, "name": "decide", "arguments": REQUEST})[
        "result"
    ]
    assert result["resultType"] == "complete" and result["isError"] is False
    assert set(result["structuredContent"]) == {"answers", "usage"}


def test_modern_version_and_metadata_errors(models: list[ServedModel]) -> None:
    server = McpServer(models)
    old = {**META, "io.modelcontextprotocol/protocolVersion": "2025-11-25"}
    error = rpc(server, "tools/list", {"_meta": old})["error"]
    assert error["code"] == UNSUPPORTED_VERSION
    assert error["data"] == {"supported": list(MODERN), "requested": "2025-11-25"}

    no_caps = {"io.modelcontextprotocol/protocolVersion": MODERN[0]}
    assert rpc(server, "tools/list", {"_meta": no_caps})["error"]["code"] == INVALID_PARAMS
    # Neither per-request metadata nor an initialize handshake.
    assert rpc(server, "tools/list", {})["error"]["code"] == INVALID_PARAMS
    # A ping is answered whatever the era.
    assert rpc(server, "ping")["result"] == {}


def test_tool_errors_and_protocol_errors(models: list[ServedModel]) -> None:
    server = McpServer(models)
    rpc(server, "initialize", {"protocolVersion": LEGACY[0]})
    unknown = rpc(server, "tools/call", {"name": "nope", "arguments": REQUEST})
    assert unknown["error"]["code"] == INVALID_PARAMS
    assert rpc(server, "resources/list", {})["error"]["code"] == METHOD_NOT_FOUND

    invalid = rpc(server, "tools/call", {"name": "decide", "arguments": {"state": "x"}})
    assert invalid["result"]["isError"] is True
    too_many = {
        "state": "x",
        "questions": {"q": {"type": "choice", "criteria": [f"o{i}" for i in range(21)]}},
    }
    refused = rpc(server, "tools/call", {"name": "decide", "arguments": too_many})
    assert refused["result"]["isError"] is True
    assert "cannot answer" in refused["result"]["content"][0]["text"]


def test_several_models_get_one_tool_each(models: list[ServedModel]) -> None:
    both = [
        ServedModel("org/model:a", models[0].package, models[0].runtime),
        ServedModel("org/model:b", models[0].package, models[0].runtime),
    ]
    server = McpServer(both)
    rpc(server, "initialize", {"protocolVersion": LEGACY[0]})
    names = [t["name"] for t in rpc(server, "tools/list", {})["result"]["tools"]]
    assert names == ["decide_org_model_a", "decide_org_model_b"]
    assert tool_name("x", many=False) == "decide"


def test_the_stdio_framing(models: list[ServedModel]) -> None:
    lines = [
        b"not json",
        b"[]",
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}).encode(),
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "initialize",
                "params": {"protocolVersion": "2025-11-25"},
            }
        ).encode(),
        b"",
    ]
    out = io.BytesIO()
    run(McpServer(models), io.BytesIO(b"\n".join(lines) + b"\n"), out)
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert replies[0]["error"]["code"] == PARSE_ERROR and replies[0]["id"] is None
    assert replies[1]["error"]["code"] == -32600
    assert replies[2]["id"] == 7 and replies[2]["result"]["protocolVersion"] == "2025-11-25"
    assert len(replies) == 3  # nothing for the notification or the blank line


def test_noulxp_mcp_over_real_stdio(toy_package: Path) -> None:  # noqa: F811
    """The command itself: only protocol messages on stdout, logs on stderr."""
    messages = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-11-25"},
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "decide", "arguments": REQUEST},
        },
    ]
    done = subprocess.run(
        [sys.executable, "-m", "noulxp", "mcp", str(toy_package), "--device", "cpu"],
        input="".join(json.dumps(m) + "\n" for m in messages),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    replies = [json.loads(line) for line in done.stdout.splitlines()]
    assert [r["id"] for r in replies] == [1, 2, 3]
    assert replies[2]["result"]["isError"] is False
    assert "ready on stdio" in done.stderr
