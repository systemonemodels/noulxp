"""The JSON Schemas of the standard (schemas/ in the repository, opendxp/schemas when installed)."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

NAMES = {
    "manifest": "odxp.schema.json",
    "template": "template.schema.json",
    "prompt": "prompt.schema.json",
    "calibration": "calibration.schema.json",
    "conformance-case": "conformance-case.schema.json",
    "request": "request.schema.json",
    "response": "response.schema.json",
    "check-report": "check-report.schema.json",
    "models": "models.schema.json",
    "error": "error.schema.json",
}


def schema_dir() -> Path:
    installed = Path(__file__).parent / "schemas"
    if installed.is_dir():
        return installed
    return Path(__file__).resolve().parents[2] / "schemas"


@cache
def schema(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((schema_dir() / NAMES[name]).read_text(encoding="utf-8"))
    return data


def errors(instance: Any, name: str) -> list[str]:
    """Every way `instance` breaks the named schema, as readable lines."""
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(schema(name))
    out = []
    for error in sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path)):
        where = "/".join(str(p) for p in error.absolute_path) or "(root)"
        out.append(f"{where}: {error.message}")
    return out
