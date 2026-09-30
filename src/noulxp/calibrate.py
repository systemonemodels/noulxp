"""noulxp calibrate: a package's temperatures fitted to labelled requests (SPEC.md 7.1).

A package's calibration is data: the temperature each question type's scores are
divided by before the softmax. Fitted to labelled requests from where a model is
used, it changes how sure the model's answers say it is, and not what the model
read (for encoder-markers packages, not which option leads either). The package
is untouched, so its conformance file still checks it; a runtime answers with the
fitted file when given it (`--calibration`).

A labelled file is JSON lines, each a request with its labels:

    {"state": ..., "questions": {...}, "labels": {"intent": "refund", "angry": true}}

A label is an option's key (a choice's name, a score's level, "true" or "false"),
or a distribution over the options ({"refund": 0.8, "cancel": 0.2}), or an answer
in the System One format (its "probabilities", or a noul's "noul"), for instance
another model's. A noul label can also be true, false or the probability that the
statement holds. Questions without a label are asked and not scored.

For each question type with labels, the temperature is the one with the least mean
KL(label || answer) over its labelled questions (the log loss, for one-hot labels),
searched over [0.01, 1000]. The report scores the answers before and after with the
four numbers decision benchmarks use: accuracy (a tie among the leading options
earns its chance), KL (answers floored at 1e-6), Brier (the sum over options of the
squared difference) and top-label ECE (10 equal bins), and the mean confidence.

What the labels are decides what the answers are calibrated to. One option per
question (what was right) fits them to how often the model is right: its mean
confidence comes to its accuracy. Distributions (another model's answers, several
annotators) fit them to that spread. Fit on requests the model was not trained on:
on data it has seen, it is right more often than it will be, and the fit says so.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from noulxp import __version__, spec
from noulxp.calibration import Calibration, Readout
from noulxp.errors import RequestError
from noulxp.request import Question, parse_questions

EPS = 1e-6
T_MIN, T_MAX = 0.01, 1000.0
GRID = 49  # log-spaced temperatures tried first, 1.27 apart
STEPS = 30  # golden-section steps between the best one's neighbours

Pairs = list[tuple[list[float], list[float]]]


class LabelError(ValueError):
    """A labelled file that cannot be read, with the line it broke on."""


@dataclass(frozen=True)
class Labelled:
    state: Any
    questions: Any
    labels: dict[str, list[float]]  # question id -> the label over its options, in their order
    line: int


def target(question: Question, label: Any) -> list[float]:
    """A label as a distribution over the question's options, in their order."""
    keys = question.keys
    if isinstance(label, dict):
        if "probabilities" in label:
            label = label["probabilities"]
        elif question.type == "noul" and "noul" in label:
            label = label["noul"]
    if question.type == "noul" and not isinstance(label, (dict, str)):
        if isinstance(label, bool):
            p = 1.0 if label else 0.0
        elif isinstance(label, (int, float)) and 0 <= label <= 1:
            p = float(label)
        else:
            raise LabelError("a noul label is true, false or the probability that it holds")
        return [1.0 - p, p]  # keys "false", "true"
    if isinstance(label, int) and not isinstance(label, bool) and question.type == "score":
        label = str(label)
    if isinstance(label, str):
        if label not in keys:
            raise LabelError(f"{label!r} is not one of its options ({', '.join(keys)})")
        return [1.0 if k == label else 0.0 for k in keys]
    if isinstance(label, dict):
        unknown = sorted(set(map(str, label)) - set(keys))
        if unknown:
            raise LabelError(f"{', '.join(map(repr, unknown))} not among its options")
        weights = [label.get(k, 0) for k in keys]
        if not all(isinstance(w, (int, float)) and not isinstance(w, bool) for w in weights):
            raise LabelError("a distribution's weights are numbers")
        if any(w < 0 for w in weights) or not sum(weights) > 0:
            raise LabelError("a distribution's weights are at least 0, and not all 0")
        total = float(sum(weights))
        return [w / total for w in weights]
    raise LabelError("a label is an option's key, a distribution over the options, or an answer")


def read_labelled(path: str | Path) -> list[Labelled]:
    """The labelled requests of a JSON-lines file."""
    out = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except ValueError as exc:
            raise LabelError(f"{path}, line {n}: not JSON ({exc})") from None
        if not isinstance(case, dict) or "questions" not in case or "labels" not in case:
            raise LabelError(f"{path}, line {n}: a line has questions and labels")
        try:
            parsed = {q.id: q for q in parse_questions(case["questions"])}
        except RequestError as exc:
            raise LabelError(f"{path}, line {n}: {exc}") from None
        if not isinstance(case["labels"], dict):
            raise LabelError(f"{path}, line {n}: labels are an object, by question id")
        labels = {}
        for qid, label in case["labels"].items():
            if qid not in parsed:
                raise LabelError(f"{path}, line {n}: there is no question {qid!r}")
            try:
                labels[qid] = target(parsed[qid], label)
            except LabelError as exc:
                raise LabelError(f"{path}, line {n}, question {qid!r}: {exc}") from None
        out.append(Labelled(case.get("state", ""), case["questions"], labels, n))
    if not out:
        raise LabelError(f"{path}: no labelled requests")
    return out


def leading(cases: list[Labelled]) -> list[Labelled]:
    """Each label replaced by its leading option (the first, in a tie)."""
    out = []
    for case in cases:
        labels = {}
        for qid, g in case.labels.items():
            top = max(range(len(g)), key=g.__getitem__)
            labels[qid] = [1.0 if i == top else 0.0 for i in range(len(g))]
        out.append(replace(case, labels=labels))
    return out


def kl(label: list[float], p: list[float]) -> float:
    return sum(g * math.log(g / max(x, EPS)) for g, x in zip(label, p, strict=True) if g > 0)


def scores(pairs: Pairs, bins: int = 10) -> dict[str, Any]:
    """(label, answer) distributions over the same options -> the benchmark's four numbers, and
    the mean confidence (the leading option's probability) to set the accuracy against."""
    correct = total_kl = brier = sure = 0.0
    buckets: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for label, p in pairs:
        top = max(range(len(label)), key=label.__getitem__)
        best = max(p)
        tied = [i for i, x in enumerate(p) if abs(x - best) < 1e-12]
        hit = 1.0 / len(tied) if top in tied else 0.0
        correct += hit
        sure += best
        total_kl += kl(label, p)
        brier += sum((x - g) ** 2 for g, x in zip(label, p, strict=True))
        buckets[min(bins - 1, int(best * bins))].append((best, hit))
    n = len(pairs)
    if not n:
        return {"decisions": 0}
    ece = sum(
        len(b) / n * abs(sum(c for c, _ in b) / len(b) - sum(h for _, h in b) / len(b))
        for b in buckets.values()
    )
    return {
        "decisions": n,
        "confidence": round(sure / n, 4),
        "accuracy": round(correct / n, 4),
        "kl": round(total_kl / n, 4),
        "brier": round(brier / n, 4),
        "ece": round(ece, 4),
    }


def answered(
    readouts: list[Readout], labelled: list[Labelled], calibration: Calibration
) -> dict[str, Pairs]:
    """(label, answer) for every labelled question, by question type."""
    by_type: dict[str, Pairs] = defaultdict(list)
    for readout, case in zip(readouts, labelled, strict=True):
        for q, p in readout(calibration):
            label = case.labels.get(q.id)
            if label is not None:
                by_type[q.type].append((label, p))
    return by_type


def summary(by_type: dict[str, Pairs]) -> dict[str, Any]:
    every = [pair for pairs in by_type.values() for pair in pairs]
    ordered = sorted(by_type, key=spec.QUESTION_TYPES.index)
    return {"all": scores(every), "by_type": {t: scores(by_type[t]) for t in ordered}}


def _losses(
    readouts: list[Readout], labelled: list[Labelled], temperatures: dict[str, float]
) -> dict[str, float]:
    """The mean KL(label || answer) of each type, at these temperatures."""
    by_type = answered(readouts, labelled, Calibration({"temperature": temperatures}))
    return {t: sum(kl(g, p) for g, p in pairs) / len(pairs) for t, pairs in by_type.items()}


def fit(readouts: list[Readout], labelled: list[Labelled], types: list[str]) -> dict[str, Any]:
    """Each type's temperature with the least mean KL: the best of a log-spaced grid, then a
    golden-section search between its neighbours. The types are searched together, each
    pass over the readouts answering every type at its own candidate. A type whose answers
    no temperature changes gets None."""
    lo, hi = math.log(T_MIN), math.log(T_MAX)
    grid = [lo + (hi - lo) * i / (GRID - 1) for i in range(GRID)]
    tried = [_losses(readouts, labelled, dict.fromkeys(types, math.exp(x))) for x in grid]

    def lowest(t: str) -> int:
        return min(range(GRID), key=lambda i: tried[i][t])

    best = {t: lowest(t) for t in types}
    a = {t: grid[max(best[t] - 1, 0)] for t in types}
    b = {t: grid[min(best[t] + 1, GRID - 1)] for t in types}
    inv = (math.sqrt(5) - 1) / 2
    c = {t: b[t] - inv * (b[t] - a[t]) for t in types}
    d = {t: a[t] + inv * (b[t] - a[t]) for t in types}
    fc = _losses(readouts, labelled, {t: math.exp(c[t]) for t in types})
    fd = _losses(readouts, labelled, {t: math.exp(d[t]) for t in types})
    for _ in range(STEPS):
        lower: dict[str, bool] = {}
        for t in types:
            lower[t] = fc[t] < fd[t]
            if lower[t]:
                b[t], d[t], fd[t] = d[t], c[t], fc[t]
                c[t] = b[t] - inv * (b[t] - a[t])
            else:
                a[t], c[t], fc[t] = c[t], d[t], fd[t]
                d[t] = a[t] + inv * (b[t] - a[t])
        got = _losses(readouts, labelled, {t: math.exp(c[t] if lower[t] else d[t]) for t in types})
        for t in types:
            if lower[t]:
                fc[t] = got[t]
            else:
                fd[t] = got[t]
    out: dict[str, Any] = {}
    for t in types:
        seen = [row[t] for row in tried]
        if max(seen) - min(seen) <= 1e-12 * max(1.0, abs(min(seen))):
            # Its answers are the same at every temperature (its options scored alike).
            out[t] = {"temperature": None, "bound": "no effect"}
            continue
        edge = "highest" if best[t] == GRID - 1 else "lowest" if best[t] == 0 else None
        out[t] = {"temperature": float(f"{math.exp((a[t] + b[t]) / 2):.4g}"), "bound": edge}
    return out


def fitted_calibration(
    package: dict[str, Any], temperatures: dict[str, float], source: dict[str, Any]
) -> dict[str, Any]:
    """The package's calibration.json with these types' temperatures in place: their
    by_option_count entries go, and everything else stays."""
    data: dict[str, Any] = {
        "standard": package.get("standard", spec.STANDARD),
        "temperature": {**(package.get("temperature") or {}), **temperatures},
    }
    buckets = [b for b in package.get("by_option_count") or [] if b.get("type") not in temperatures]
    if buckets:
        data["by_option_count"] = buckets
    data["source"] = source
    return data


def calibrate(
    package_dir: str | Path,
    labels: str | Path,
    *,
    test: str | Path | None = None,
    hard: bool = False,
    device: str = "auto",
    threads: int | None = None,
    log: Any = print,
    **options: Any,
) -> dict[str, Any]:
    """Fit the package's temperatures to `labels`; score before and after (and on `test`).

    With `hard`, the fit is to each label's leading option: to how often the model is
    right, even from labels that are distributions. The scores are against the labels
    as given."""
    from noulxp.package import open_package, sha256_file
    from noulxp.runtime import load

    labelled = read_labelled(labels)
    tested = read_labelled(test) if test else []
    package = open_package(package_dir)
    runtime = load(package, device=device, threads=threads, **options)
    try:
        log(f"reading {len(labelled) + len(tested)} labelled requests with {package.name} ...")
        began = time.perf_counter()
        readouts = runtime.readouts([(c.state, c.questions) for c in labelled])
        test_readouts = runtime.readouts([(c.state, c.questions) for c in tested])
        read_s = time.perf_counter() - began
        own = runtime.calibration
        described = runtime.describe()
    finally:
        runtime.close()

    def usable(
        results: list[Readout | Exception], cases: list[Labelled]
    ) -> tuple[list[Readout], list[Labelled], list[int]]:
        kept = [(r, c) for r, c in zip(results, cases, strict=True) if not isinstance(r, Exception)]
        refused = [c.line for r, c in zip(results, cases, strict=True) if isinstance(r, Exception)]
        return [r for r, _ in kept], [c for _, c in kept], refused

    fit_readouts, fit_cases, refused = usable(readouts, labelled)
    before = answered(fit_readouts, fit_cases, own)
    types = sorted(before, key=spec.QUESTION_TYPES.index)
    if not types:
        raise LabelError(f"{labels}: no labelled question was answered")
    log(f"fitting {', '.join(types)} ...")
    fitted = fit(fit_readouts, leading(fit_cases) if hard else fit_cases, types)
    temperatures = {t: fitted[t]["temperature"] for t in types if fitted[t]["temperature"]}
    own_data = own.data
    source = {
        "fitted_by": f"noulxp {__version__} calibrate",
        "package": package.name,
        "labels": {
            "file": Path(labels).name,
            "sha256": sha256_file(Path(labels)),
            "requests": len(fit_cases),
            "decisions": {t: len(before[t]) for t in types},
            "as": "each label's leading option" if hard else "given",
        },
        "objective": "the least mean KL(label || answer), per question type",
        "package_calibration": {
            k: own_data[k] for k in ("temperature", "by_option_count") if k in own_data
        },
    }
    data = fitted_calibration(own_data, temperatures, source)
    new = Calibration(data)
    report: dict[str, Any] = {
        "package": {"name": package.name, "profile": package.profile},
        "runtime": described,
        "read_s": round(read_s, 1),
        "calibration": data,
        "fitted": {
            t: {
                "decisions": len(before[t]),
                "package_temperature": own_data.get("temperature", {}).get(t, 1.0),
                "package_buckets": any(
                    b.get("type") == t for b in own_data.get("by_option_count") or []
                ),
                **fitted[t],
            }
            for t in types
        },
        "labelled": {
            "requests": len(fit_cases),
            "refused": refused,
            "before": summary(before),
            "after": summary(answered(fit_readouts, fit_cases, new)),
        },
    }
    if tested:
        test_fit, test_cases, test_refused = usable(test_readouts, tested)
        report["test"] = {
            "file": Path(test).name if test else None,
            "requests": len(test_cases),
            "refused": test_refused,
            "before": summary(answered(test_fit, test_cases, own)),
            "after": summary(answered(test_fit, test_cases, new)),
        }
    return report


def table(report: dict[str, Any]) -> str:
    """The report as a table: each type's temperature and scores, before -> after."""
    lines = []

    def block(title: str, part: dict[str, Any]) -> None:
        before, after = part["before"], part["after"]
        lines.append(
            f"{title}: {part['requests']:,} requests"
            + (f", {len(part['refused'])} refused" if part["refused"] else "")
        )
        head = ["", "decisions", "temperature", "confidence", "accuracy", "KL", "Brier", "ECE"]
        grid = [head]
        rows = [(t, before["by_type"][t], after["by_type"][t]) for t in before["by_type"]]
        rows.append(("all", before["all"], after["all"]))
        for name, x, y in rows:
            info = report["fitted"].get(name)
            temp = ""
            if info:
                was = "by count" if info["package_buckets"] else f"{info['package_temperature']:g}"
                temp = f"{was} -> {info['temperature']:g}" if info["temperature"] else was
            right = f"{x['accuracy']:.4g}"
            if y["accuracy"] != x["accuracy"]:
                right += f" -> {y['accuracy']:.4g}"
            grid.append(
                [name, f"{x['decisions']:,}", temp]
                + [f"{x[k]:.4g} -> {y[k]:.4g}" for k in ("confidence",)]
                + [right]
                + [f"{x[k]:.4g} -> {y[k]:.4g}" for k in ("kl", "brier", "ece")]
            )
        widths = [max(len(row[i]) for row in grid) for i in range(len(head))]
        for row in grid:
            cells = [
                cell.rjust(w) if i == 1 else cell.ljust(w)
                for i, (cell, w) in enumerate(zip(row, widths, strict=True))
            ]
            lines.append("  " + "  ".join(cells).rstrip())

    lines.append(f"{report['package']['name']} ({report['runtime'].get('backend', '')})")
    block("fitted on", report["labelled"])
    if "test" in report:
        block("test", report["test"])
    for t, info in report["fitted"].items():
        if info["bound"] == "highest":
            lines.append(
                f"  {t}: the best temperature is the highest tried: on these labels its answers"
                " carry little information, and fitted they say so (close to even)"
            )
        elif info["bound"] == "lowest":
            lines.append(f"  {t}: the best temperature is the lowest tried")
        elif info["bound"] == "no effect":
            lines.append(f"  {t}: no temperature changes its answers here; the package's is kept")
    return "\n".join(lines)
