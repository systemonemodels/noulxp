"""Random System One requests from a small vocabulary, deterministic by seed."""

from __future__ import annotations

import random
from typing import Any

from conftest import WORDS

UNKNOWN = ["zorbly", "quux", "flim", "blarg"]


def words(rng: random.Random, n: int) -> str:
    pool = WORDS + UNKNOWN + ["?", ",", ".", "(", ")"]
    return " ".join(rng.choice(pool) for _ in range(n))


def maybe_marker(rng: random.Random, text: str, marker: str, p: float = 0.05) -> str:
    return f"{text} {marker} {text}" if rng.random() < p else text


def question(rng: random.Random, kind: str, marker: str, max_options: int = 12) -> dict[str, Any]:
    long = rng.random() < 0.15
    instructions = maybe_marker(rng, words(rng, rng.randint(1, 60 if long else 12)), marker)
    if kind == "choice":
        n = rng.randint(2, max_options)
        names = [f"opt{i}{rng.choice(['', 'x', 'y'])}" for i in range(n)]
        if rng.random() < 0.3:
            return {"type": "choice", "instructions": instructions, "criteria": names}
        criteria = {}
        for name in names:
            r = rng.random()
            if r < 0.3:
                criteria[name] = None
            elif r < 0.35:
                criteria[name] = ""
            else:
                size = rng.randint(1, 70) if rng.random() < 0.1 else rng.randint(1, 8)
                criteria[name] = maybe_marker(rng, words(rng, size), marker)
        return {"type": "choice", "instructions": instructions, "criteria": criteria}
    if kind == "score":
        n = rng.randint(2, min(10, max_options))
        levels = [
            (f"{i}: " if rng.random() < 0.2 else "") + words(rng, rng.randint(1, 6))
            for i in range(n)
        ]
        return {"type": "score", "instructions": instructions, "criteria": levels}
    spec: dict[str, Any] = {"type": "noul", "instructions": instructions}
    r = rng.random()
    if r < 0.2:
        spec["criteria"] = {"true": words(rng, 3)}
    elif r < 0.4:
        spec["criteria"] = {"false": words(rng, 2), "true": words(rng, 4)}
    elif r < 0.5:
        spec["instructions"] = ""
        spec["criteria"] = {"true": words(rng, 3)}
    return spec


def request(rng: random.Random, marker: str, max_options: int = 12) -> dict[str, Any]:
    n = rng.randint(1, 3)
    questions = {
        f"q{i}": question(rng, rng.choice(["choice", "score", "noul"]), marker, max_options)
        for i in range(n)
    }
    size = rng.choice([0, 1, 5, 20, 60, 150])
    state = maybe_marker(rng, words(rng, size), marker)
    if rng.random() < 0.1:
        state += "\n"
    return {"state": state, "questions": questions}
