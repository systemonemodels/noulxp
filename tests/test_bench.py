"""`noulxp bench`: the toy package in this process, and through a real server."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from noulxp import cli
from noulxp.bench import bench_http, bench_package, percentile, summarize, table
from noulxp.server import serve
from noulxp.serving import load_models
from test_conformance import TOY_REQUESTS, toy_package  # noqa: F401 - a fixture

REQUESTS = [row["request"] for row in TOY_REQUESTS]
QUESTIONS = sum(len(r["questions"]) for r in REQUESTS)


def test_percentiles_interpolate() -> None:
    assert percentile([], 50) is None
    assert percentile([5.0], 99) == 5.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert percentile([1.0, 2.0, 3.0, 4.0], 100) == 4.0


def test_a_level_counts_decisions_and_prices_them() -> None:
    level = summarize([10.0, 20.0], 6, {"503": 1}, 2.0, concurrency=4, usd_per_hour=0.72)
    assert level["requests"] == 3 and level["answered"] == 2
    assert level["decisions_per_s"] == 3.0 and level["requests_per_s"] == 1.0
    # 0.72 dollars an hour at 3 decisions a second: 10,800 an hour, so $0.0667 per 1,000.
    assert level["usd_per_1k_decisions"] == pytest.approx(0.066667, abs=1e-6)
    assert level["latency_ms"]["p50"] == 15.0


def test_a_package_in_this_process(toy_package: Path) -> None:  # noqa: F811
    report = bench_package(
        toy_package, REQUESTS, device="cpu", rounds=2, warmup=1, log=lambda _: None
    )
    [level] = report["levels"]
    assert report["target"]["kind"] == "package"
    assert level["answered"] == 2 * len(REQUESTS)
    assert level["decisions"] == 2 * QUESTIONS
    assert level["decisions_per_s"] > 0 and level["latency_ms"]["p50"] is not None
    assert "decisions/s" in table(report)


def test_a_server_at_two_levels(toy_package: Path) -> None:  # noqa: F811
    models = load_models([toy_package], device="cpu", log=lambda *_: None)
    server = serve(models, port=0, quiet=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    try:
        report = bench_http(
            f"http://{host}:{port}",
            REQUESTS,
            concurrency=[1, 2],
            duration_s=0.5,
            warmup=1,
            usd_per_hour=1.0,
            log=lambda _: None,
        )
    finally:
        server.shutdown()
        for model in models:
            model.close()
    assert [level["concurrency"] for level in report["levels"]] == [1, 2]
    for level in report["levels"]:
        assert level["answered"] > 0 and level["errors"] == {}
        assert level["decisions"] >= level["answered"]
        assert level["usd_per_1k_decisions"] > 0
    # The server describes what it holds.
    assert report["target"]["models"]["data"][0]["object"] == "model"


def test_the_command_writes_a_report(
    toy_package: Path,  # noqa: F811
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    requests = tmp_path / "requests.jsonl"
    requests.write_text("\n".join(json.dumps(row) for row in TOY_REQUESTS) + "\n")
    out = tmp_path / "bench.json"
    code = cli.main(
        [
            "bench",
            str(toy_package),
            "--requests",
            str(requests),
            "--device",
            "cpu",
            "--warmup",
            "0",
            "--report",
            str(out),
        ]
    )
    assert code == 0
    assert json.loads(out.read_text())["levels"][0]["answered"] == len(REQUESTS)
    assert "decisions/s" in capsys.readouterr().out


def test_the_client_turns_nagle_off(toy_package: Path) -> None:  # noqa: F811
    import socket

    from noulxp.bench import _Connection

    loaded = load_models([toy_package], device="cpu", log=lambda *_: None)
    server = serve(loaded, port=0, quiet=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        host, port = server.server_address[:2]
        connection = _Connection(f"http://{host}:{port}/v1/systemone", None, 30.0)
        status, answer, _ = connection.post(TOY_REQUESTS[0]["request"])
        assert status == 200 and answer is not None
        assert connection.conn is not None and connection.conn.sock is not None
        assert connection.conn.sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)
        connection.close()
    finally:
        server.shutdown()
        for model in loaded:
            model.close()
