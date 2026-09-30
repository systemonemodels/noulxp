"""Shared steps of the encoder-markers converters: sample batches, verification, packaging."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from noulxp.calibration import Calibration
from noulxp.profiles.encoder_markers import Template, encode_question, pack
from noulxp.request import parse_questions
from noulxp.tokens import Tokens

# Requests the exporters trace and verify the graph with: several lengths (inside and
# beyond the 128-token attention window), option counts and all three question types.
_LONG = (
    "The customer wrote again about order 1182: the card was charged twice, the duplicate "
    "has not been refunded, and they want an update today before they escalate. "
)
SAMPLE_REQUESTS: list[dict[str, Any]] = [
    {
        "state": "I was charged twice this month and support never replied.",
        "questions": {
            "intent": {
                "type": "choice",
                "instructions": "What does the customer want?",
                "criteria": {"refund": "money back", "cancel": None, "track": "delivery status"},
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this?",
                "criteria": ["not urgent", "somewhat urgent", "very urgent"],
            },
        },
    },
    {
        "state": _LONG * 6,
        "questions": {
            "angry": {"type": "noul", "instructions": "Is the customer angry?"},
            "team": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": ["billing", "shipping", "accounts", "technical", "sales", "other"],
            },
        },
    },
    {
        "state": _LONG * 30,
        "questions": {
            "topic": {
                "type": "choice",
                "instructions": "Which topic is this about?",
                "criteria": [f"topic {i}" for i in range(12)],
            }
        },
    },
]


def sample_batches(template: Template, tokens: Tokens) -> list[dict[str, Any]]:
    pad = tokens.id(template.special["pad"])
    batches = []
    for request in SAMPLE_REQUESTS:
        encoded = []
        for q in parse_questions(request["questions"]):
            try:
                encoded.append(encode_question(template, tokens, q, request["state"]))
            except ValueError:
                continue  # a strict template may refuse the longest sample
        if encoded:
            batches.append(pack(encoded, pad))
    return batches


def trace_batch(batches: list[dict[str, Any]]) -> dict[str, Any]:
    """A batch with more than one row, token and option, so no dimension is specialised."""
    for batch in batches:
        n, t = batch["input_ids"].shape
        if n > 1 and t > 1 and batch["marker_positions"].shape[1] > 1:
            return batch
    return batches[0]


def calibration_ok(data: dict[str, Any]) -> dict[str, Any]:
    Calibration(data)  # raises if invalid
    return data


def tokenizer_path(checkpoint: Path) -> Path:
    for candidate in (checkpoint / "tokenizer" / "tokenizer.json", checkpoint / "tokenizer.json"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no tokenizer.json in {checkpoint}")


def tokenizer_config(checkpoint: Path) -> dict[str, Any]:
    import json

    for candidate in (
        checkpoint / "tokenizer" / "tokenizer_config.json",
        checkpoint / "tokenizer_config.json",
    ):
        if candidate.is_file():
            data = json.loads(candidate.read_text())
            return data if isinstance(data, dict) else {}
    return {}
