"""Filling the declarative templates, and turning request values into text.

Templates are data, never code: a placeholder is `{name}` with a lowercase
name, it is replaced in a single pass (so text from a request that happens to
contain "{name}" is never expanded), and an unknown placeholder stays as it is.
"""

from __future__ import annotations

import json
import re
from typing import Any

_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")


def fill(template: str, **values: str) -> str:
    """Replace each `{key}` in `template` whose key is given; leave the rest."""

    def substitute(match: re.Match[str]) -> str:
        key = match.group(1)
        return values[key] if key in values else match.group(0)

    return _PLACEHOLDER.sub(substitute, template)


def placeholders(template: str) -> set[str]:
    return set(_PLACEHOLDER.findall(template))


def as_text(value: Any) -> str:
    """A criterion (description, level) as text: strings as they are, anything else as JSON.

    JSON uses ", " and ": " as separators and keeps non-ASCII characters, which
    is how Laya and Decider both render structured criteria.
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def annotate_indices(value: Any, minimum: int) -> Any:
    """Decider's state annotation: arrays of `minimum` or more items get an "_index" per item."""
    if isinstance(value, list):
        if len(value) >= minimum:
            return [
                {"_index": i, **annotate_indices(v, minimum)}
                if isinstance(v, dict)
                else {"_index": i, "value": annotate_indices(v, minimum)}
                for i, v in enumerate(value)
            ]
        return [annotate_indices(v, minimum) for v in value]
    if isinstance(value, dict):
        return {k: annotate_indices(v, minimum) for k, v in value.items()}
    return value


def state_text(state: Any, index_arrays_from: int | None = None) -> str:
    """The state as the model reads it: a string as it is, JSON otherwise."""
    if isinstance(state, str):
        return state
    if index_arrays_from:
        state = annotate_indices(state, index_arrays_from)
    return json.dumps(state, ensure_ascii=False)
