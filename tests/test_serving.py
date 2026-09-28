"""The HTTP binding (SPEC.md 13) on the toy package, through a real server on a free port."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from opendxp import schemas, spec
from opendxp.server import DecisionServer, is_loopback, serve
from opendxp.serving import ServedModel, load_models
from test_conformance import TOY_REQUESTS, toy_package  # noqa: F401 - a fixture

REQUEST = TOY_REQUESTS[0]["request"]
TOO_MANY = {
    "state": "x",
    "questions": {"q": {"type": "choice", "criteria": [f"o{i}" for i in range(21)]}},
}


@pytest.fixture()
def models(toy_package: Path) -> Iterator[list[ServedModel]]:  # noqa: F811
    loaded = load_models([toy_package], device="cpu", log=lambda *_: None)
    yield loaded
    for model in loaded:
        model.close()


def start(models: list[ServedModel], **options: Any) -> DecisionServer:
    server = serve(models, port=0, quiet=True, **options)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def call(
    server: DecisionServer,
    path: str,
    body: Any = None,
    headers: dict[str, str] | None = None,
    method: str | None = None,
) -> tuple[int, dict[str, str], Any]:
    host, port = server.server_address[:2]
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        f"http://{host}:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
            return response.status, dict(response.headers), json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        return exc.code, dict(exc.headers), json.loads(raw) if raw else None


def test_discovery_and_an_answer(models: list[ServedModel]) -> None:
    server = start(models)
    try:
        status, _, health = call(server, "/healthz")
        assert status == 200 and health == {"ok": True, "models": 1}

        status, headers, listed = call(server, spec.HTTP_MODELS_PATH)
        assert status == 200 and schemas.errors(listed, "models") == []
        assert [m["id"] for m in listed["data"]] == ["test/toy"]
        assert listed["data"][0]["profile"] == "encoder-markers"
        assert headers["OpenDXP-Version"] == spec.PROTOCOL_VERSION

        status, headers, answer = call(server, spec.HTTP_DECIDE_PATH, REQUEST)
        assert status == 200, answer
        assert answer["model"] == "test/toy"
        assert schemas.errors(answer, "response") == []
        # The server answers exactly what the runtime does.
        direct = models[0].runtime.predict(REQUEST["state"], REQUEST["questions"])
        assert answer["answers"] == direct["answers"]
        assert answer["latency_ms"] >= 0
    finally:
        server.shutdown()


def test_errors_have_their_status_and_type(models: list[ServedModel]) -> None:
    server = start(models)
    try:
        cases = [
            ({"state": "x"}, 400, "invalid_request"),
            ({**REQUEST, "model": "nope"}, 404, "model_not_found"),
            (TOO_MANY, 422, "request_refused"),
        ]
        for body, status, kind in cases:
            got, _, error = call(server, spec.HTTP_DECIDE_PATH, body)
            assert (got, error["error"]["type"]) == (status, kind), error
            assert schemas.errors(error, "error") == []
        got, _, error = call(server, "/v1/nothing", REQUEST)
        assert got == 404 and error["error"]["type"] == "not_found"
    finally:
        server.shutdown()


def test_a_request_over_the_size_limit_is_refused(models: list[ServedModel]) -> None:
    server = start(models)
    try:
        body = {**REQUEST, "state": "x" * spec.MAX_REQUEST_BYTES}
        status, _, error = call(server, spec.HTTP_DECIDE_PATH, body)
        assert status == 413 and error["error"]["type"] == "request_too_large"
    finally:
        server.shutdown()


def test_a_server_with_several_models_needs_the_model_named(
    models: list[ServedModel],
) -> None:
    both = [
        ServedModel("a", models[0].package, models[0].runtime),
        ServedModel("b", models[0].package, models[0].runtime),
    ]
    server = start(both)
    try:
        status, _, error = call(server, spec.HTTP_DECIDE_PATH, REQUEST)
        assert status == 400 and "a, b" in error["error"]["message"]
        status, _, answer = call(server, spec.HTTP_DECIDE_PATH, {**REQUEST, "model": "b"})
        assert status == 200 and answer["model"] == "b"
    finally:
        server.shutdown()


def test_a_token_guards_everything_but_health(models: list[ServedModel]) -> None:
    server = start(models, token="s3cret")
    try:
        assert call(server, "/healthz")[0] == 200
        assert call(server, spec.HTTP_MODELS_PATH)[0] == 401
        assert call(server, spec.HTTP_DECIDE_PATH, REQUEST)[0] == 401
        wrong = {"Authorization": "Bearer nope"}
        assert call(server, spec.HTTP_MODELS_PATH, headers=wrong)[0] == 401
        right = {"Authorization": "Bearer s3cret"}
        assert call(server, spec.HTTP_MODELS_PATH, headers=right)[0] == 200
        assert call(server, spec.HTTP_DECIDE_PATH, REQUEST, headers=right)[0] == 200
    finally:
        server.shutdown()


def test_browsers_are_let_in_only_from_named_origins(models: list[ServedModel]) -> None:
    server = start(models, cors=["http://localhost:3000"])
    try:
        allowed = {"Origin": "http://localhost:3000"}
        _, headers, _ = call(server, spec.HTTP_MODELS_PATH, headers=allowed)
        assert headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"
        _, headers, _ = call(server, spec.HTTP_MODELS_PATH, headers={"Origin": "http://evil"})
        assert "Access-Control-Allow-Origin" not in headers
        status, headers, _ = call(server, spec.HTTP_DECIDE_PATH, headers=allowed, method="OPTIONS")
        assert status == 204 and "POST" in headers["Access-Control-Allow-Methods"]
    finally:
        server.shutdown()


def test_it_will_not_listen_beyond_this_machine_without_a_token(
    models: list[ServedModel],
) -> None:
    assert is_loopback("127.0.0.1") and is_loopback("::1") and is_loopback("localhost")
    assert not is_loopback("0.0.0.0") and not is_loopback("192.168.1.5")
    with pytest.raises(ValueError, match="without a token"):
        serve(models, host="0.0.0.0", port=0)
