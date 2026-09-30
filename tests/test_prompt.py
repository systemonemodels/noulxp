"""causal-letters prompt construction against Decider's own code (tests/oracles.py)."""

from __future__ import annotations

import random

import pytest

from generators import request
from noulxp.errors import PackageError, RequestError
from noulxp.export import decider as decider_export
from noulxp.profiles.causal_letters import Builder, Prompt, unique_tokens
from noulxp.request import parse_questions
from oracles import DeciderPrompter, HFStyle

MAX_OPTIONS = 36  # A..Z and AA..AJ are single tokens of the test tokenizer


@pytest.fixture(scope="module")
def builder(tokens):  # type: ignore[no-untyped-def]
    labels = decider_export.labels(tokens, MAX_OPTIONS)
    cfg = {"isolated_levels": True, "max_state_tokens": 100000, "max_options": MAX_OPTIONS}
    return Builder(Prompt(decider_export.prompt(cfg, labels)), tokens)


def rows_of(builder: Builder, state, questions):  # type: ignore[no-untyped-def]
    context = builder.context(state)
    out = []
    for q in parse_questions(questions):
        _, rows = builder.rows(q)
        out.append([context + builder.question_piece(t, o) for t, o in rows])
    return out


def test_labels_follow_decider(tokens, builder):
    native = DeciderPrompter(HFStyle(tokens), MAX_OPTIONS)
    assert builder.label_ids == native.label_ids
    assert builder.prompt.labels[:10] == list("ABCDEFGHIJ")


def test_prompt_rows_reproduce_decider(tokens, builder):
    native = DeciderPrompter(HFStyle(tokens), MAX_OPTIONS)
    rng = random.Random(11)
    compared = 0
    for _ in range(300):
        req = request(rng, "<mask>", max_options=24)
        assert rows_of(builder, req["state"], req["questions"]) == native.rows(
            req["state"], req["questions"], True
        )
        compared += 1
    assert compared == 300


def test_wide_layout_is_split_at_the_label(tokens, builder):
    options = [f"opt{i}" for i in range(12)]
    piece = builder.question_piece("which", options)
    open_ids = tokens.encode("\n(")
    first = tokens.encode("\n\nQuestion: which\nOptions:")
    assert piece[: len(first)] == first
    assert piece[len(first) : len(first) + len(open_ids) + 1] == [*open_ids, builder.label_ids[0]]


def test_isolated_levels_strip_numbers(builder):
    q = parse_questions(
        {"s": {"type": "score", "instructions": "how", "criteria": ["0: low", "high"]}}
    )[0]
    kind, rows = builder.rows(q)
    assert kind == "isolated"
    assert rows[0] == ("how\nProposed answer: low\nDoes the proposed answer fit?", ["no", "yes"])
    assert rows[1][0].endswith("Proposed answer: high\nDoes the proposed answer fit?")


def test_instructions_fallback_and_requirement(builder):
    noul = parse_questions({"n": {"type": "noul", "instructions": "", "criteria": {"true": "x"}}})[
        0
    ]
    assert builder.rows(noul)[1][0][0] == "Which answer fits the context?"
    choice = parse_questions({"c": {"type": "choice", "instructions": "", "criteria": ["a", "b"]}})[
        0
    ]
    with pytest.raises(RequestError):
        builder.rows(choice)


def test_unique_tokens_counts_shared_prefixes_once():
    assert unique_tokens(10, [[1, 2, 3]]) == 13
    assert unique_tokens(10, [[1, 2, 3], [1, 2, 4, 5]]) == 10 + 2 + 1 + 2


def test_prompt_validation(tokens):
    labels = decider_export.labels(tokens, 12)
    good = decider_export.prompt({"isolated_levels": False}, labels)
    assert Prompt(good).score["mode"] == "list"
    for broken in (
        {**good, "question": {**good["question"], "option": "\n{text}"}},
        {**good, "slot": "first"},
        {**good, "layouts": [{"max_options": 40, "encoding": "joined"}]},
        {**good, "context": {"template": "no state here"}},
    ):
        with pytest.raises(PackageError):
            Prompt(broken)
    with pytest.raises(PackageError):
        Builder(
            Prompt(
                {
                    **good,
                    "labels": ["A", "not-a-single-token-label"],
                    "layouts": [{"max_options": 2, "encoding": "joined"}],
                }
            ),
            tokens,
        )


def test_llama_cpp_logging_is_silenced_once_for_the_process() -> None:
    # llama.cpp keeps one callback for the whole process: a callback freed with
    # one runtime would be called by the next, so it is installed once and kept.
    from noulxp.profiles.causal_letters import _silence

    class FakeLlama:
        def __init__(self) -> None:
            self.installed: list[object] = []

        def llama_log_callback(self, fn: object) -> object:
            return ("callback", fn)

        def llama_log_set(self, callback: object, data: object) -> None:
            self.installed.append(callback)

    lc = FakeLlama()
    first = _silence(lc)
    assert _silence(lc) is first
    assert lc.installed == [first]
