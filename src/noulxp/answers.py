"""From scores to System One answers: the calibrated softmax, confidence, answer objects."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from noulxp.errors import NoulXPError
from noulxp.request import Question
from noulxp.spec import DECIMALS


def softmax(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    """softmax(z / T) in float64, over the logits given (the valid options only)."""
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    if not np.all(np.isfinite(z)):
        raise NoulXPError("the model returned non-finite scores")
    e = np.exp(z - z.max())
    return (e / e.sum()).tolist()


def normalise(values: Sequence[float]) -> list[float]:
    total = float(sum(values))
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [float(v) / total for v in values]


def confidence(rule: str, p: Sequence[float]) -> float | None:
    """The confidence rules a package can declare (SPEC.md, "Answers")."""
    n = len(p)
    if rule == "none":
        return None
    if rule == "max-probability":
        return float(max(p))
    if rule == "entropy":
        if n < 2:
            return 1.0
        h = -sum(x * math.log(min(max(x, 1e-12), 1.0)) for x in p)
        return min(1.0, max(0.0, 1.0 - h / math.log(n)))
    if rule == "typesafe":
        if n < 2:
            return 1.0
        return min(1.0, max(0.0, (n * max(p) - 1) / (n - 1)))
    if rule == "typesafe-ordinal":
        if n < 2:
            return 1.0
        k = max(range(n), key=p.__getitem__)
        spread = sum(x * abs(i - k) for i, x in enumerate(p))
        uniform = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
        return min(1.0, max(0.0, 1.0 - spread / uniform))
    raise NoulXPError(f"unknown confidence rule {rule!r}")


def argmax(p: Sequence[float]) -> int:
    """The first index of the largest value."""
    return max(range(len(p)), key=p.__getitem__)


def answer(
    question: Question,
    p: Sequence[float],
    rules: dict[str, str] | None = None,
    decimals: int | None = DECIMALS,
) -> dict[str, Any]:
    """One answer in the System One format; probabilities rounded to `decimals` if set."""

    def r(x: float) -> float:
        return round(float(x), decimals) if decimals is not None else float(x)

    rule = (rules or {}).get(question.type, "none")
    conf = confidence(rule, p)
    keys = question.keys
    best = argmax(p)
    out: dict[str, Any]
    if question.type == "choice":
        out = {
            "type": "choice",
            "choice": keys[best],
            "probabilities": {k: r(x) for k, x in zip(keys, p, strict=True)},
        }
    elif question.type == "score":
        out = {
            "type": "score",
            "score": r(sum(i * x for i, x in enumerate(p))),
            "legend": question.legend(),
            "probabilities": {k: r(x) for k, x in zip(keys, p, strict=True)},
        }
    else:
        out = {"type": "noul", "noul": r(p[1])}
    if conf is not None:
        out["confidence"] = r(conf)
    return out
