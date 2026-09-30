"""Conformance: what a model's own code answers, and whether a runtime reproduces it.

conformance.jsonl holds one case per line:

    {"id": "...", "request": {"state": ..., "questions": {...}},
     "expected": {qid: {"type": "choice", "probabilities": {key: p}}}}
    {"id": "...", "request": {...}, "error": {"type": "...", "message": "..."}}

A runtime passes a case when, for every question, the probabilities it computes
are within TOLERANCE of the expected ones and its argmax is the expected argmax
(or ties with it within TIE_MARGIN), or when the case expects an error and the
runtime refuses the request too. Noul answers compare over (false, true).
"""

from __future__ import annotations

import gc
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from noulxp import __version__
from noulxp.errors import NoulXPError, RequestError
from noulxp.package import MANIFEST, Package, find_manifest, open_package, sha256_file
from noulxp.request import parse_questions
from noulxp.spec import MIN_CASES, TIE_MARGIN, TOLERANCE

CONFORMANCE_FILE = "conformance.jsonl"
DATA = Path(__file__).parent / "data"
DEFAULT_REQUESTS = DATA / "requests-0.1.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    cases = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if line.strip():
                try:
                    cases.append(json.loads(line))
                except ValueError as exc:
                    raise NoulXPError(f"{path.name}:{n} is not JSON: {exc}") from exc
    return cases


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def expected_from_answer(answer: dict[str, Any]) -> dict[str, Any]:
    """The probabilities a System One answer states, as a conformance expectation."""
    kind = answer.get("type")
    if kind in ("choice", "score"):
        probabilities = {str(k): float(v) for k, v in answer["probabilities"].items()}
    elif kind == "noul":
        p = float(answer["noul"])
        probabilities = {"false": round(1.0 - p, 10), "true": p}
    else:
        raise NoulXPError(f"not a System One answer: {answer!r}")
    return {"type": kind, "probabilities": probabilities}


def generate(
    package_dir: Path,
    native: Any,
    requests: list[dict[str, Any]],
    *,
    log: Any = print,
) -> dict[str, Any]:
    """Run the model's own code on every request and write conformance.jsonl into the package."""
    package_dir = Path(package_dir)
    rows = []
    errors = 0
    started = time.perf_counter()
    for i, item in enumerate(requests, 1):
        request = item["request"]
        row: dict[str, Any] = {"id": item["id"], "request": request}
        try:
            result = native.predict(request.get("state", ""), request["questions"])
            row["expected"] = {qid: expected_from_answer(a) for qid, a in result["answers"].items()}
        except Exception as exc:
            row["error"] = {"type": type(exc).__name__, "message": str(exc)[:500]}
            errors += 1
        for key in ("tags",):
            if key in item:
                row[key] = item[key]
        rows.append(row)
        if i % 10 == 0:
            log(f"  {i}/{len(requests)} cases ({time.perf_counter() - started:.0f} s)")
    path = package_dir / CONFORMANCE_FILE
    write_jsonl(path, rows)
    questions = sum(len(r.get("expected", {})) for r in rows)
    summary = {
        "path": CONFORMANCE_FILE,
        "sha256": sha256_file(path),
        "cases": len(rows),
        "questions": questions,
        "errors": errors,
        "tolerance": {"max_abs_dp": TOLERANCE, "tie_margin": TIE_MARGIN},
        "generated_by": {
            **native.provenance,
            "tool": f"noulxp {__version__}",
            "seconds": round(time.perf_counter() - started, 1),
        },
    }
    manifest_path = find_manifest(package_dir) or package_dir / MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["conformance"] = summary
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return summary


def compare_question(
    expected: dict[str, Any], keys: list[str], observed: list[float]
) -> dict[str, Any]:
    want = expected["probabilities"]
    if list(want) != keys:
        return {"ok": False, "reason": f"option keys differ: expected {list(want)}, got {keys}"}
    dp = max(abs(float(want[k]) - p) for k, p in zip(keys, observed, strict=True))
    top = max(want.values())
    tied = {k for k, v in want.items() if v >= top - TIE_MARGIN}
    got = keys[max(range(len(observed)), key=observed.__getitem__)]
    argmax_ok = got in tied
    return {
        "ok": dp <= TOLERANCE and argmax_ok,
        "dp": dp,
        "argmax_ok": argmax_ok,
        "expected_argmax": max(want, key=want.__getitem__),
        "observed_argmax": got,
    }


def coverage(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """What a conformance file exercises, against the minimums in SPEC.md 9.2."""
    types: Counter[str] = Counter()
    options: set[int] = set()
    languages: set[str] = set()
    longest = 0
    for case in cases:
        request = case["request"]
        state = request.get("state", "")
        longest = max(longest, len(state if isinstance(state, str) else json.dumps(state)))
        for tag in case.get("tags", []):
            if tag.startswith("lang:"):
                languages.add(tag[5:])
        for q in parse_questions(request["questions"]):
            types[q.type] += 1
            if q.type == "choice":
                options.add(len(q.options))
    checks = {
        "cases": len(cases) >= MIN_CASES,
        "question_types": all(types[t] >= 5 for t in ("choice", "score", "noul")),
        "choice_2_to_10_options": set(range(2, 11)) <= options or max(options, default=0) >= 10,
        "more_than_10_options": max(options, default=0) > 10,
        "languages": len(languages) >= 3,
        "long_state": longest >= 2000,
    }
    return {
        "cases": len(cases),
        "questions_by_type": dict(types),
        "choice_option_counts": sorted(options),
        "languages": sorted(languages),
        "longest_state_chars": longest,
        "checks": checks,
        "ok": all(checks.values()),
    }


def check(
    package_dir: Path,
    *,
    device: str = "cpu",
    threads: int | None = None,
    hashes: bool = True,
    log: Any = print,
    **options: Any,
) -> dict[str, Any]:
    """Run the package's conformance file through the reference runtime."""
    from noulxp.runtime import load

    package = open_package(package_dir)
    problems = package.verify(hashes=hashes)
    runtime = load(package, device=device, threads=threads, **options)
    try:
        return replay(package, runtime, device=device, threads=threads, problems=problems)
    finally:
        runtime.close()
        gc.collect()  # release native sessions now, not during interpreter shutdown


def replay(
    package: Package,
    runtime: Any,
    *,
    device: str,
    threads: int | None = None,
    problems: list[str] | None = None,
) -> dict[str, Any]:
    """The package's conformance file through a runtime that is already loaded.

    An engine can check a package as it serves it (its device, precision and settings)
    without loading it twice. The runtime is left open. A runtime given another
    calibration (SPEC.md 7.1) is checked at its package's, which the file was recorded at.
    """
    from noulxp.providers import describe_machine

    problems = list(problems or [])
    entry = package.manifest.get("conformance") or {"path": CONFORMANCE_FILE}
    cases = read_jsonl(package.resolve(entry["path"]))
    described = runtime.describe()

    own = getattr(runtime, "package_calibration", None)
    at = {"calibration": own} if own is not None else {}

    order = list(cases)
    if getattr(runtime, "static", False):
        # A fixed-shape backend compiles one graph per shape bucket: visit each bucket once.
        order.sort(key=lambda c: _bucket(runtime, c))
    warmed: set[Any] = set()

    results, times, per_question, tokens = [], [], [], []
    worst = 0.0
    total_dp, n_dp = 0.0, 0
    agree = asked = 0
    errors_expected = errors_matched = 0
    for case in order:
        request = case["request"]
        state, questions = request.get("state", ""), request["questions"]
        key = _bucket(runtime, case) if getattr(runtime, "static", False) else None
        if "expected" in case and key not in warmed:  # warm-up (and compilation), not timed
            warmed.add(key)
            try:
                runtime.distributions(state, questions, **at)
            except RequestError:
                pass
        began = time.perf_counter()
        observed: Any
        try:
            observed, usage = runtime.distributions(state, questions, **at)
            refused = None
        except RequestError as exc:
            observed, usage, refused = None, None, exc
        elapsed = (time.perf_counter() - began) * 1000
        record: dict[str, Any] = {"id": case["id"]}
        if "error" in case:
            errors_expected += 1
            record["expected_error"] = case["error"]["message"]
            record["ok"] = refused is not None
            errors_matched += int(refused is not None)
            if refused is None:
                record["reason"] = (
                    "the model's own code refused this request; this runtime answered"
                )
        elif refused is not None:
            record["ok"] = False
            record["reason"] = f"refused: {refused}"
        else:
            times.append(elapsed)
            per_question.append(elapsed / max(1, len(observed)))
            tokens.append(usage["input_tokens"])
            ok = True
            details = {}
            for q, p in observed:
                result = compare_question(case["expected"][q.id], q.keys, p)
                details[q.id] = result
                if "dp" in result:
                    worst = max(worst, result["dp"])
                    total_dp += result["dp"]
                    n_dp += 1
                    agree += int(result["argmax_ok"])
                    asked += 1
                ok = ok and result["ok"]
            record["ok"] = ok
            record["max_dp"] = max((d.get("dp", 1.0) for d in details.values()), default=0.0)
            if not ok:
                record["questions"] = {k: v for k, v in details.items() if not v["ok"]}
        results.append(record)
    load_ms = float(getattr(runtime, "load_ms", 0.0))

    failures = [r for r in results if not r["ok"]]
    cov = coverage(cases)
    passed = not failures and not problems
    report = {
        "standard": package.manifest["standard"],
        "package": {
            "name": package.name,
            "profile": package.profile,
            # The folder's name, not where it sits: reports are published next to badges.
            "path": package.root.resolve().name,
            "conformance_sha256": entry.get("sha256"),
            "weights_sha256": package.entry("weights").get("sha256"),
        },
        "runtime": {
            "implementation": f"noulxp {__version__}",
            **described,
            "device": device,
            "threads": threads,
        },
        "machine": describe_machine(),
        "tolerance": {"max_abs_dp": TOLERANCE, "tie_margin": TIE_MARGIN},
        "cases": len(cases),
        "cases_passed": len(cases) - len(failures),
        "questions": asked,
        "max_abs_dp": worst,
        "mean_abs_dp": total_dp / n_dp if n_dp else 0.0,
        "argmax_agreement": {"agree": agree, "total": asked},
        "errors": {"expected": errors_expected, "matched": errors_matched},
        "latency_ms": {
            "load": round(load_ms, 1),
            "request_median": _round(statistics.median(times)) if times else None,
            "request_p90": _round(_quantile(times, 0.9)) if times else None,
            "request_mean": _round(statistics.fmean(times)) if times else None,
            "question_median": _round(statistics.median(per_question)) if times else None,
            "input_tokens_mean": _round(statistics.fmean(tokens)) if tokens else None,
        },
        "package_problems": problems,
        "coverage": cov,
        "passed": passed,
        "compatible": passed and cov["ok"],
        "failures": failures[:25],
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return report


def _bucket(runtime: Any, case: dict[str, Any]) -> tuple[int, int]:
    """The shape bucket a static-shape runtime will run this case at ((0, 0) if refused)."""
    request = case["request"]
    try:
        encoded = runtime.encode(request.get("state", ""), parse_questions(request["questions"]))
    except (RequestError, NoulXPError):
        return (0, 0)
    bucket: tuple[int, int] = runtime.bucket(
        max(len(e.ids) for e in encoded), max(len(e.markers) for e in encoded)
    )
    return bucket


def _round(x: float) -> float:
    return round(float(x), 2)


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]
