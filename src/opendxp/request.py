"""The System One request, checked and normalised once for every profile.

A request is {"state": ..., "questions": {id: {"type", "instructions", "criteria"}}}.
Each question becomes a `Question` with its options in the caller's order; each
option has the key the answer uses, a name, and a description (None when the
request gave none, which templates render with their "bare" form):

    choice  criteria {name: description | null} or [name, ...]   key = name
    score   criteria [level description, ...], lowest first       key = "0", "1", ...
    noul    criteria null or {"false": ..., "true": ...}          key = "false", "true"

Descriptions that are not strings are rendered as JSON (see text.as_text).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from opendxp.errors import RequestError
from opendxp.spec import QUESTION_TYPES
from opendxp.text import as_text

NOUL_KEYS = ("false", "true")


@dataclass(frozen=True)
class Option:
    key: str
    name: str
    description: str | None
    index: int


@dataclass(frozen=True)
class Question:
    id: str
    type: str
    instructions: str
    options: tuple[Option, ...]

    @property
    def keys(self) -> list[str]:
        return [o.key for o in self.options]

    def legend(self) -> dict[str, str]:
        """Score answers carry the level texts under their keys."""
        return {o.key: o.description or "" for o in self.options}


def _description(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return as_text(value)


def parse_question(qid: str, spec: Any) -> Question:
    if not isinstance(spec, dict):
        raise RequestError(f"question {qid!r} is not an object")
    kind = spec.get("type")
    if kind not in QUESTION_TYPES:
        raise RequestError(f"question {qid!r}: type must be one of {', '.join(QUESTION_TYPES)}")
    instructions = spec.get("instructions", "")
    if instructions is None:
        instructions = ""
    if not isinstance(instructions, str):
        raise RequestError(f"question {qid!r}: instructions must be text")
    criteria = spec.get("criteria")
    options: list[Option] = []
    if kind == "choice":
        if isinstance(criteria, list):
            names = criteria
            if not all(isinstance(n, str) and n for n in names):
                raise RequestError(f"question {qid!r}: choice options must be non-empty strings")
            if len(set(names)) != len(names):
                raise RequestError(f"question {qid!r}: choice options must be unique")
            options = [Option(n, n, None, i) for i, n in enumerate(names)]
        elif isinstance(criteria, dict):
            if not all(isinstance(n, str) and n for n in criteria):
                raise RequestError(f"question {qid!r}: choice names must be non-empty strings")
            options = [
                Option(n, n, _description(d), i) for i, (n, d) in enumerate(criteria.items())
            ]
        else:
            raise RequestError(f"question {qid!r}: choice criteria must be an object or a list")
    elif kind == "score":
        if not isinstance(criteria, list):
            raise RequestError(f"question {qid!r}: score criteria must be a list, lowest first")
        options = [Option(str(i), str(i), _description(d), i) for i, d in enumerate(criteria)]
    else:
        given = {} if criteria is None else criteria
        if not isinstance(given, dict) or not set(given) <= set(NOUL_KEYS):
            raise RequestError(
                f'question {qid!r}: noul criteria may only describe "false" and "true"'
            )
        options = [Option(k, k, _description(given.get(k)), i) for i, k in enumerate(NOUL_KEYS)]
    if len(options) < 2:
        raise RequestError(f"question {qid!r} needs at least 2 options")
    return Question(str(qid), kind, instructions, tuple(options))


def parse_questions(questions: Any) -> list[Question]:
    if not isinstance(questions, dict) or not questions:
        raise RequestError("questions must be a non-empty object")
    return [parse_question(str(qid), spec) for qid, spec in questions.items()]


def parse_request(request: Any) -> tuple[Any, list[Question]]:
    if not isinstance(request, dict) or "questions" not in request:
        raise RequestError('a request is {"state": ..., "questions": {...}}')
    state = request.get("state", "")
    if state is None:
        state = ""
    return state, parse_questions(request["questions"])
