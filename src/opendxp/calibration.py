"""calibration.json: the temperature each answer is divided by before the softmax.

A temperature is looked up by question type and option count: the first entry
of `by_option_count` whose type matches and whose [min, max] range holds the
count wins; otherwise the per-type temperature; otherwise 1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opendxp.errors import PackageError
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


def _positive(value: Any, where: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise PackageError(f"calibration: {where} is not a number") from None
    if not number > 0 or number == float("inf"):
        raise PackageError(f"calibration: {where} must be a positive finite number")
    return number
