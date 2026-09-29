"""`opendxp bench`: how fast a package or an OpenDXP server answers, in a report anyone can compare.

Two kinds of target:

- a package directory: the reference runtime in this process, one request at a time;
- the URL of any server that speaks the HTTP binding (SPEC.md 11): `opendxp serve`, an
  engine, a hosted API. Each concurrency level runs that many clients at once for a while.

The requests are the OpenDXP request set unless another is given, the same 52 requests the
conformance files use. A decision is one question answered, so a request with three questions
is three decisions. The report names the machine and the target and gives, for each level,
requests and decisions per second, latency percentiles and errors, and, with a price per hour,
what 1,000 decisions cost. Speed means nothing without exactness: run `opendxp check` on the
same machine and device, and publish both.
"""

from __future__ import annotations

import itertools
import json
import os
import statistics
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opendxp import __version__
from opendxp.conformance import DEFAULT_REQUESTS, read_jsonl

Log = Callable[[str], None]
PERCENTILES = (50, 90, 95, 99)


def load_requests(path: Path | None = None, *, limit: int | None = None) -> list[dict[str, Any]]:
    """The requests to send: {"state", "questions"} each, from a requests JSONL."""
    rows = read_jsonl(path or DEFAULT_REQUESTS)
    requests = [row["request"] if "request" in row else row for row in rows]
    return requests[:limit] if limit else requests


def percentile(values: list[float], q: float) -> float | None:
    """The q-th percentile (0 to 100) by linear interpolation; None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q / 100
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def summarize(
    latencies_ms: list[float],
    decisions: int,
    errors: dict[str, int],
    elapsed_s: float,
    *,
    concurrency: int,
    usd_per_hour: float | None = None,
) -> dict[str, Any]:
    """One level's numbers."""
    ok = len(latencies_ms)
    per_second = decisions / elapsed_s if elapsed_s > 0 else 0.0
    level: dict[str, Any] = {
        "concurrency": concurrency,
        "requests": ok + sum(errors.values()),
        "answered": ok,
        "errors": dict(errors),
        "elapsed_s": round(elapsed_s, 3),
        "requests_per_s": round(ok / elapsed_s, 3) if elapsed_s > 0 else 0.0,
        "decisions": decisions,
        "decisions_per_s": round(per_second, 3),
        "latency_ms": {
            **{f"p{q}": _round(percentile(latencies_ms, q)) for q in PERCENTILES},
            "mean": _round(statistics.fmean(latencies_ms)) if latencies_ms else None,
            "max": _round(max(latencies_ms)) if latencies_ms else None,
        },
    }
    if usd_per_hour is not None and per_second > 0:
        level["usd_per_1k_decisions"] = round(usd_per_hour / (per_second * 3600) * 1000, 6)
    return level


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 2)


# --- a package, in this process ---------------------------------------------------------


def bench_package(
    package_dir: Path,
    requests: list[dict[str, Any]],
    *,
    device: str = "auto",
    threads: int | None = None,
    rounds: int = 1,
    warmup: int = 3,
    usd_per_hour: float | None = None,
    log: Log = print,
    **options: Any,
) -> dict[str, Any]:
    """The reference runtime answering the requests one at a time, `rounds` times over.

    `options` go to the runtime (for example `batch_rows` for a causal-letters package).
    """
    from opendxp.package import open_package
    from opendxp.providers import describe_machine
    from opendxp.runtime import load

    package = open_package(package_dir)
    started = time.perf_counter()
    runtime = load(package, device=device, threads=threads, **options)
    load_ms = (time.perf_counter() - started) * 1000
    for request in requests[:warmup]:
        runtime.predict(request.get("state", ""), request["questions"])

    latencies: list[float] = []
    errors: dict[str, int] = {}
    decisions = 0
    tokens = 0
    t0 = time.perf_counter()
    for i in range(rounds):
        for request in requests:
            began = time.perf_counter()
            try:
                result = runtime.predict(request.get("state", ""), request["questions"])
            except (ValueError, KeyError) as exc:
                key = type(exc).__name__
                errors[key] = errors.get(key, 0) + 1
                continue
            latencies.append((time.perf_counter() - began) * 1000)
            decisions += len(result.get("answers") or {})
            tokens += int((result.get("usage") or {}).get("input_tokens") or 0)
        log(f"  round {i + 1}/{rounds}: {len(latencies)} answered")
    elapsed = time.perf_counter() - t0

    level = summarize(
        latencies, decisions, errors, elapsed, concurrency=1, usd_per_hour=usd_per_hour
    )
    level["input_tokens_per_s"] = round(tokens / elapsed, 1) if elapsed > 0 else 0.0
    return _report(
        target={
            "kind": "package",
            "name": package.manifest.get("name"),
            "profile": package.profile,
            "path": str(package_dir),
            "runtime": runtime.describe() if hasattr(runtime, "describe") else None,
            "load_ms": round(load_ms, 1),
        },
        machine=describe_machine(),
        requests=requests,
        rounds=rounds,
        warmup=warmup,
        usd_per_hour=usd_per_hour,
        levels=[level],
    )


# --- a server, over HTTP --------------------------------------------------------------


def _post(
    url: str, body: dict[str, Any], token: str | None, timeout: float
) -> tuple[int, dict[str, Any] | None, float | None]:
    """(status, answer, the Retry-After the server asked for)."""
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else None, None
    except urllib.error.HTTPError as exc:
        exc.read()
        return exc.code, None, _seconds(exc.headers.get("Retry-After"))
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        return 0, {"error": type(exc).__name__}, None


def _seconds(value: str | None) -> float | None:
    try:
        return float(value) if value else None
    except ValueError:
        return None


def _get(url: str, token: str | None, timeout: float = 10.0) -> Any:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=timeout
        ) as response:
            return json.loads(response.read() or b"null")
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None


def _level(
    endpoint: str,
    bodies: list[dict[str, Any]],
    token: str | None,
    timeout_s: float,
    clients: int,
    duration_s: float,
    backoff_s: float = 0.25,
) -> tuple[list[float], int, dict[str, int], float]:
    """`clients` clients sending the bodies round and round until the time is up.

    A client told the server is busy (429 or 503) waits as a real one would: the Retry-After it was
    given, at most one second, or `backoff_s`, before its next request. Without that, an overloaded
    server is measured answering its own rejections.
    """
    latencies: list[float] = []
    errors: dict[str, int] = {}
    decisions = 0
    lock = threading.Lock()
    order = itertools.cycle(range(len(bodies)))
    deadline = time.perf_counter() + duration_s

    def client() -> None:
        nonlocal decisions
        while time.perf_counter() < deadline:
            with lock:
                body = bodies[next(order)]
            began = time.perf_counter()
            status, answer, retry_after = _post(endpoint, body, token, timeout_s)
            took = (time.perf_counter() - began) * 1000
            with lock:
                if status == 200 and answer is not None:
                    latencies.append(took)
                    decisions += len(answer.get("answers") or {})
                else:
                    key = str(status or (answer or {}).get("error", "no answer"))
                    errors[key] = errors.get(key, 0) + 1
            if status in (429, 503):
                time.sleep(min(retry_after if retry_after is not None else backoff_s, 1.0))

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=clients) as pool:
        for future in [pool.submit(client) for _ in range(clients)]:
            future.result()
    return latencies, decisions, errors, time.perf_counter() - t0


def bench_http(
    url: str,
    requests: list[dict[str, Any]],
    *,
    model: str | None = None,
    extra: dict[str, Any] | None = None,
    token: str | None = None,
    concurrency: list[int] | None = None,
    duration_s: float = 30.0,
    warmup: int = 3,
    timeout_s: float = 120.0,
    usd_per_hour: float | None = None,
    log: Log = print,
) -> dict[str, Any]:
    """Every level in turn: that many clients sending the requests round and round for a while."""
    from opendxp.providers import describe_machine

    base = url.rstrip("/")
    endpoint = base if base.endswith("/v1/systemone") else f"{base}/v1/systemone"
    root = endpoint.removesuffix("/v1/systemone")
    fields = {**({"model": model} if model else {}), **(extra or {})}
    bodies = [{**fields, **request} for request in requests]

    for body in bodies[:warmup]:
        _post(endpoint, body, token, timeout_s)

    levels = []
    for clients in concurrency or [1, 4, 16]:
        latencies, decisions, errors, elapsed = _level(
            endpoint, bodies, token, timeout_s, clients, duration_s
        )
        level = summarize(
            latencies, decisions, errors, elapsed, concurrency=clients, usd_per_hour=usd_per_hour
        )
        levels.append(level)
        log(
            f"  {clients:>3} clients: {level['decisions_per_s']:.1f} decisions/s, "
            f"p50 {level['latency_ms']['p50']} ms, errors {sum(errors.values())}"
        )

    return _report(
        target={
            "kind": "server",
            "url": endpoint,
            "model": model,
            "models": _get(f"{root}/v1/models", token),
        },
        machine={"client": describe_machine()},
        requests=requests,
        rounds=None,
        warmup=warmup,
        usd_per_hour=usd_per_hour,
        levels=levels,
        duration_s=duration_s,
    )


def _report(
    *,
    target: dict[str, Any],
    machine: dict[str, Any],
    requests: list[dict[str, Any]],
    rounds: int | None,
    warmup: int,
    usd_per_hour: float | None,
    levels: list[dict[str, Any]],
    duration_s: float | None = None,
) -> dict[str, Any]:
    return {
        "tool": f"opendxp {__version__}",
        "bench_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target": target,
        "machine": machine,
        "requests": {
            "count": len(requests),
            "questions": sum(len(r.get("questions") or {}) for r in requests),
        },
        "settings": {
            "rounds": rounds,
            "warmup": warmup,
            "duration_s": duration_s,
            "usd_per_hour": usd_per_hour,
        },
        "levels": levels,
    }


def table(report: dict[str, Any]) -> str:
    """The report's levels as a plain-text table."""
    head = (
        f"{'clients':>7} {'req/s':>8} {'decisions/s':>12} {'p50 ms':>9} {'p95 ms':>9} {'errors':>7}"
    )
    cost = report["settings"].get("usd_per_hour") is not None
    if cost:
        head += f" {'$/1k decisions':>15}"
    lines = [head]
    for level in report["levels"]:
        lat = level["latency_ms"]
        row = (
            f"{level['concurrency']:>7} {level['requests_per_s']:>8.2f} "
            f"{level['decisions_per_s']:>12.2f} {lat['p50'] or 0:>9.1f} {lat['p95'] or 0:>9.1f} "
            f"{sum(level['errors'].values()):>7}"
        )
        if cost:
            price = level.get("usd_per_1k_decisions")
            row += f" {price if price is not None else float('nan'):>15.5f}"
        lines.append(row)
    return "\n".join(lines)


def token_from_env() -> str | None:
    return os.environ.get("OPENDXP_TOKEN") or None
