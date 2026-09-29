"""calibration.json: the temperature each answer is divided by before the softmax.

A temperature is looked up by question type and option count: the first entry
of `by_option_count` whose type matches and whose [min, max] range holds the
count wins; otherwise the per-type temperature; otherwise 1.

A runtime can answer with another calibration.json than its package's (SPEC.md
7.1), for example one `opendxp calibrate` fitted to labelled requests.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opendxp.errors import PackageError
from opendxp.request import Question
from opendxp.spec import QUESTION_TYPES


@dataclass(frozen=True)
class Bucket:
    type: str
    min: int
    max: int | None
    temperature: float

    def holds(self, qtype: str, count: int) -> bool:
        return qtype == self.type and count >= self.min and (self.max is None or count <= self.max)


class Calibration:
    def __init__(self, data: dict[str, Any] | None = None) -> None:
        data = data or {}
        self.data = data
        self.by_type: dict[str, float] = {}
        for qtype, value in (data.get("temperature") or {}).items():
            if qtype not in QUESTION_TYPES:
                raise PackageError(f"calibration: unknown question type {qtype!r}")
            self.by_type[qtype] = _positive(value, f"temperature.{qtype}")
        self.buckets: list[Bucket] = []
        for i, entry in enumerate(data.get("by_option_count") or []):
            qtype = entry.get("type")
            if qtype not in QUESTION_TYPES:
                raise PackageError(f"calibration: by_option_count[{i}] has no valid type")
            low, high = entry.get("min", 2), entry.get("max")
            if not isinstance(low, int) or (high is not None and not isinstance(high, int)):
                raise PackageError(f"calibration: by_option_count[{i}] min/max must be integers")
            if high is not None and high < low:
                raise PackageError(f"calibration: by_option_count[{i}] has max < min")
            self.buckets.append(
                Bucket(qtype, low, high, _positive(entry.get("temperature"), f"bucket {i}"))
            )

    @classmethod
    def from_file(cls, path: Path) -> Calibration:
        return cls(json.loads(Path(path).read_text()))

    def temperature(self, qtype: str, count: int) -> float:
        for bucket in self.buckets:
            if bucket.holds(qtype, count):
                return bucket.temperature
        return self.by_type.get(qtype, 1.0)


# A request read once, answered at any calibration: its questions and their probabilities.
Readout = Callable[[Calibration], list[tuple[Question, list[float]]]]


def read_calibration(path: str | Path) -> tuple[dict[str, Any], str]:
    """A calibration.json given to a runtime (SPEC.md 7.1), checked, and its sha256."""
    from opendxp import schemas
    from opendxp.package import sha256_file

    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PackageError(f"calibration {path}: {exc}") from None
    problems = schemas.errors(data, "calibration")
    if problems:
        raise PackageError(f"calibration {path}: " + "; ".join(problems[:3]))
    Calibration(data)  # the checks the schema cannot make
    return data, sha256_file(Path(path))


def replace(runtime: Any, path: str | Path) -> None:
    """Make a loaded runtime answer with another calibration.json (SPEC.md 7.1).

    The package's own stays on the runtime as `package_calibration`, for its
    conformance file, which was recorded with it."""
    data, digest = read_calibration(path)
    runtime.package_calibration = runtime.calibration
    runtime.calibration = Calibration(data)
    runtime.calibration_replaced = {"sha256": digest, "temperature": data.get("temperature")}


def described(runtime: Any) -> Any:
    """How a runtime's report names its calibration: "package", or the file it was given."""
    return getattr(runtime, "calibration_replaced", None) or "package"


def _positive(value: Any, where: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise PackageError(f"calibration: {where} is not a number") from None
    if not number > 0 or number == float("inf"):
        raise PackageError(f"calibration: {where} must be a positive finite number")
    return number
