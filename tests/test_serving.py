"""The HTTP binding (SPEC.md 11) on the toy package, through a real server on a free port."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from noulxp import schemas, spec
from noulxp.errors import RequestError
from noulxp.server import DecisionServer, Handler, is_loopback, serve
from noulxp.serving import Batcher, ServedModel, ServingError, load_models
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
        assert headers["NoulXP-Version"] == spec.PROTOCOL_VERSION

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


def test_an_answer_leaves_in_one_write_without_nagle() -> None:
    # Otherwise, on Linux, every answer on a kept-alive connection waited ~40 ms for
    # the client's delayed ACK of its headers before its body was sent.
    assert Handler.disable_nagle_algorithm and Handler.wbufsize != 0
    assert DecisionServer.request_queue_size >= 64


def same(a: Any, b: Any) -> bool:
    """Equal, but for floating-point rounding in probabilities."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) or isinstance(b, float):
        return abs(a - b) <= 1e-3
    return bool(a == b)


def test_a_batching_server_answers_as_the_runtime_does_alone(
    toy_package: Path,  # noqa: F811
) -> None:
    loaded = load_models([toy_package], device="cpu", log=lambda *_: None, batch=32)
    assert loaded[0].batcher is not None
    runtime = loaded[0].runtime
    bodies = [case["request"] for case in TOY_REQUESTS] + [TOO_MANY]
    alone: list[Any] = []
    for body in bodies:  # before the server starts: its thread is the runtime's only caller
        try:
            alone.append(runtime.predict(body["state"], body["questions"])["answers"])
        except RequestError:
            alone.append(422)
    server = start(loaded)
    try:
        with ThreadPoolExecutor(16) as pool:
            got = list(pool.map(lambda b: call(server, spec.HTTP_DECIDE_PATH, b), bodies * 8))
        for expected, (status, _, answer) in zip(alone * 8, got, strict=True):
            if expected == 422:
                assert status == 422 and answer["error"]["type"] == "request_refused"
            else:
                assert status == 200 and same(answer["answers"], expected), answer
    finally:
        server.shutdown()
        for model in loaded:
            model.close()
    assert not loaded[0].batcher.thread.is_alive()


class Recorder:
    """A runtime that answers each request with its state, noting each pass's size.

    The first pass waits for `hold`, so requests sent meanwhile queue behind it."""

    def __init__(self) -> None:
        self.passes: list[int] = []
        self.hold = threading.Event()
        self.reading = threading.Event()

    def _pass(self, size: int) -> None:
        self.passes.append(size)
        self.reading.set()
        self.hold.wait(10)

    def _answer(self, state: str) -> dict[str, Any]:
        if state == "fail":
            raise RuntimeError("this request broke the model")
        return {"answers": {"q": state}, "usage": {"input_tokens": 1, "output_tokens": 0}}

    def predict(self, state: str, questions: Any) -> dict[str, Any]:
        self._pass(1)
        return self._answer(state)

    def predict_many(self, items: list[tuple[str, Any]]) -> list[Any]:
        self._pass(len(items))
        if any(state == "fail" for state, _ in items):
            raise RuntimeError("the pass failed")
        return [RequestError("refused") if s == "refuse" else self._answer(s) for s, _ in items]


def held(limit: int) -> tuple[Recorder, Batcher, Any]:
    """A batcher whose first pass, of request "a", is under way."""
    runtime = Recorder()
    batcher = Batcher(runtime, limit)
    first = batcher.submit("a", {})
    assert runtime.reading.wait(5)
    return runtime, batcher, first


def test_requests_that_wait_are_read_together_up_to_the_limit() -> None:
    runtime, batcher, first = held(3)
    try:
        rest = [batcher.submit(s, {}) for s in "bcdef"]
        runtime.hold.set()
        assert first.result(5)["answers"] == {"q": "a"}
        assert [f.result(5)["answers"]["q"] for f in rest] == list("bcdef")
        assert runtime.passes == [1, 3, 2]
    finally:
        runtime.hold.set()
        batcher.close()


def test_each_request_in_a_pass_gets_its_own_error() -> None:
    runtime, batcher, first = held(32)
    try:
        refused, answered = batcher.submit("refuse", {}), batcher.submit("b", {})
        runtime.hold.set()
        first.result(5)
        with pytest.raises(RequestError):
            refused.result(5)
        assert answered.result(5)["answers"] == {"q": "b"}
    finally:
        runtime.hold.set()
        batcher.close()


def test_a_pass_that_fails_is_read_again_one_by_one() -> None:
    runtime, batcher, first = held(32)
    try:
        broken, answered = batcher.submit("fail", {}), batcher.submit("b", {})
        runtime.hold.set()
        first.result(5)
        with pytest.raises(RuntimeError, match="this request"):
            broken.result(5)
        assert answered.result(5)["answers"] == {"q": "b"}
        assert runtime.passes == [1, 2, 1, 1]
    finally:
        runtime.hold.set()
        batcher.close()


def test_closing_answers_what_waits_and_turns_away_what_comes_after() -> None:
    runtime, batcher, first = held(32)
    waiting = batcher.submit("b", {})
    closing = threading.Thread(target=batcher.close)
    closing.start()
    runtime.hold.set()
    closing.join(5)
    assert not batcher.thread.is_alive()
    assert first.result(5)["answers"] == {"q": "a"}
    assert waiting.result(5)["answers"] == {"q": "b"}
    with pytest.raises(ServingError) as refused:
        batcher.submit("c", {}).result(5)
    assert (refused.value.status, refused.value.kind) == (503, "model_not_ready")
    batcher.close()  # twice is fine


def test_a_request_that_finds_the_model_idle_is_read_in_its_own_thread() -> None:
    runtime, batcher, first = held(32)  # the batcher's thread is reading "a"
    readers: dict[str, str] = {}
    predict = runtime.predict

    def noted(state: str, questions: Any) -> dict[str, Any]:
        readers[state] = threading.current_thread().name
        return predict(state, questions)

    runtime.predict = noted  # type: ignore[method-assign]
    try:
        # Busy: "b" waits, and the batcher's thread reads it after "a".
        busy = threading.Thread(
            target=lambda: readers.setdefault("b-answer", batcher.answer("b", {}))
        )
        busy.start()
        while batcher.queue.empty():
            threading.Event().wait(0.01)
        runtime.hold.set()
        busy.join(5)
        assert first.result(5)["answers"] == {"q": "a"}
        assert readers["b"] == batcher.thread.name
        # Idle: "c" is read here, in this thread.
        assert batcher.answer("c", {})["answers"] == {"q": "c"}
        assert readers["c"] == threading.current_thread().name
        assert runtime.passes == [1, 1, 1]
    finally:
        runtime.hold.set()
        batcher.close()
