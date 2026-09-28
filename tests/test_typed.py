"""Typed layouts and rotations (SPEC.md 6.7, 6.8) against AnyJev's own code.

anyjev 0.2.0 is numpy-only at its core, so these tests run its real
build_prompt, render_chat, render_state, rotation math and Decider, on a fake
model whose logits are a hash of the row it reads. The package side is what
`opendxp export anyjev` writes, run by the reference runtime's typed path.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest

anyjev = pytest.importorskip("anyjev")

from anyjev import Decider  # noqa: E402
from anyjev.calibrate.contextual import apply_contextual, content_free_prior  # noqa: E402
from anyjev.calibrate.permute import cyclic_shifts, marginalize  # noqa: E402
from anyjev.readout import (  # noqa: E402
    DEFAULT_SYSTEM,
    build_prompt,
    label_ids_for_perm,
    render_chat,
    resolve_labels,
)
from anyjev.state import render_state as anyjev_state  # noqa: E402

from conftest import WORDS  # noqa: E402
from generators import UNKNOWN, request  # noqa: E402
from opendxp import rotations, schemas  # noqa: E402
from opendxp.calibration import Calibration  # noqa: E402
from opendxp.errors import PackageError, RequestError  # noqa: E402
from opendxp.export import anyjev as exporter  # noqa: E402
from opendxp.native import anyjev as native  # noqa: E402
from opendxp.profiles.causal_letters import CausalLettersRuntime, Prompt  # noqa: E402
from opendxp.profiles.typed import TypedBuilder, trie_tokens  # noqa: E402
from opendxp.request import parse_questions  # noqa: E402
from opendxp.text import render_state  # noqa: E402
from opendxp.tokens import Tokens  # noqa: E402

SPECIALS = ["[UNK]", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
PROMPT_WORDS = (
    "State Question Options Answer Pick Yes No You Reply with the letter number only level that "
    "applies levels are ordered or empty a decision function will be given state and one question "
    "answer label no words punctuation explanation system user assistant opt x y MASK N"
).split()


def build_tokenizer(path: Path) -> Path:
    """Words, single letters and single digits: like Qwen, "10" is two tokens."""
    import string

    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab: dict[str, int] = {}
    pieces = [*SPECIALS, *string.ascii_uppercase, *string.digits, *PROMPT_WORDS, *WORDS]
    for token in pieces + list(":.,?!()[]{}/-'\"#"):
        vocab.setdefault(token, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Whitespace(), pre_tokenizers.Digits(individual_digits=True)]
    )
    tok.add_special_tokens(SPECIALS[1:])
    tok.save(str(path))
    return path


class ChatTokenizer:
    """What anyjev reads from a transformers tokenizer, with a Qwen-style chat template."""

    chat_template = "qwen-style"

    def __init__(self, tokens: Tokens) -> None:
        self.t = tokens

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.t.encode(text)

    def apply_chat_template(
        self, messages: list[dict[str, str]], tokenize: bool, add_generation_prompt: bool, **kw: Any
    ) -> str:
        assert not tokenize and add_generation_prompt and kw.get("enable_thinking") is False
        system, user = messages[0]["content"], messages[1]["content"]
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )


def fake_logits(ids: list[int], vocab: int) -> np.ndarray:
    """A deterministic model: the logits at a row's end are drawn from a hash of the row."""
    digest = hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little")).normal(0.0, 2.5, vocab)


class FakeBackend:
    """anyjev's backend protocol over the fake model: log-softmax at the last position."""

    name = "fake"

    def __init__(self, tokens: Tokens) -> None:
        self.tokenizer = ChatTokenizer(tokens)
        self.vocab = tokens.backend.get_vocab_size()

    def next_token_logprobs(self, prompts: list[str], token_ids: list[list[int]]) -> list[Any]:
        out = []
        for text, ids in zip(prompts, token_ids, strict=True):
            z = fake_logits(self.tokenizer.encode(text), self.vocab)
            lp = z - (z.max() + np.log(np.exp(z - z.max()).sum()))
            out.append(lp[list(ids)])
        return out


@pytest.fixture(scope="module")
def tokens(tmp_path_factory: pytest.TempPathFactory) -> Tokens:
    return Tokens(build_tokenizer(tmp_path_factory.mktemp("anyjev") / "tokenizer.json"))


@pytest.fixture(scope="module")
def hf(tokens: Tokens) -> ChatTokenizer:
    return ChatTokenizer(tokens)


def package_prompt(
    tokens: Tokens, hf: ChatTokenizer, *, prior: str = "none", canonical: bool = False
) -> dict[str, Any]:
    template = exporter.chat_template(hf, DEFAULT_SYSTEM)
    return exporter.prompt(tokens, template, DEFAULT_SYSTEM, canonical=canonical, prior=prior)


def runtime(tokens: Tokens, data: dict[str, Any]) -> CausalLettersRuntime:
    """The reference runtime's typed path with the fake model in place of llama.cpp."""
    rt = object.__new__(CausalLettersRuntime)
    rt.prompt = Prompt(data)
    rt.tokens = tokens
    rt.builder = TypedBuilder(rt.prompt, tokens)
    rt.calibration = Calibration(exporter.calibration())
    rt.rules = {}
    rt.limits = {"max_tokens": 40960, "max_options": 26, "max_levels": 10}
    vocab = tokens.backend.get_vocab_size()
    rt.slot_logits = lambda ids, label_ids: fake_logits(ids, vocab)[np.asarray(label_ids)]  # type: ignore[method-assign]
    return rt


STATES: list[Any] = [
    "",
    None,
    "the customer wants a refund",
    "line one\nline two\n",
    {"order": {"id": "#{123}", "items": ["a", "b"]}, "note": "très urgent", "n": 3},
    [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi, how can I help?"}],
    [{"role": "user"}, {"content": "no role"}],
    [],
    {},
    0,
]


def anyjev_perms(q: Any, canonical: bool) -> list[list[int]]:
    """anyjev.decider.Decider._perms at L0 with every shift read."""
    if q.ordered:
        return [list(range(q.k))]
    if q.kind == "noul":
        return [[0, 1], [1, 0]]
    shifts = cyclic_shifts(q.k)
    if not canonical:
        return shifts
    canon = sorted(range(q.k), key=lambda i: (str(q.options[i]), i))
    return [[canon[i] for i in perm] for perm in shifts]


def from_anyjev(kind: str, perm: list[int]) -> list[int]:
    """anyjev lists a noul as (Yes, No); a System One noul is (false, true)."""
    return [1 - i for i in perm] if kind == "noul" else list(perm)


def test_labels_follow_anyjev(tokens: Tokens, hf: ChatTokenizer) -> None:
    builder = TypedBuilder(Prompt(package_prompt(tokens, hf)), tokens)
    layouts = builder.layouts.types
    for k in range(2, 27):
        q = anyjev.Question.choice("which", [f"o{i}" for i in range(k)])
        labels, ids = resolve_labels(hf, q)
        which = layouts["choice"].layout(k)
        assert list(layouts["choice"].layouts[which].labels[:k]) == labels
        assert builder.ids[("choice", which)][:k] == ids
    for k in range(2, 11):
        q = anyjev.Question.score("how", levels=[f"l{i}" for i in range(k)])
        labels, ids = resolve_labels(hf, q)
        which = layouts["score"].layout(k)
        assert list(layouts["score"].layouts[which].labels[:k]) == labels
        assert builder.ids[("score", which)][:k] == ids
    assert resolve_labels(hf, anyjev.Question.score("how", levels=list("abcdefghij")))[0][0] == "A"
    labels, ids = resolve_labels(hf, anyjev.Question.noul("is it"))
    assert list(layouts["noul"].layouts[0].labels) == labels[::-1]  # (false, true)
    assert builder.ids[("noul", 0)] == ids[::-1]


def test_states_render_as_anyjev_renders_them(tokens: Tokens, hf: ChatTokenizer) -> None:
    spec = package_prompt(tokens, hf)["state"]
    for state in STATES:
        expected = anyjev_state(state) or "(empty)"  # build_prompt: "(empty)"
        assert render_state(state, spec) == expected, state


def test_rows_are_anyjev_prompts(tokens: Tokens, hf: ChatTokenizer) -> None:
    rng = random.Random(7)
    compared = refused = 0
    for canonical in (False, True):
        builder = TypedBuilder(Prompt(package_prompt(tokens, hf, canonical=canonical)), tokens)
        for n in range(150):
            req = request(rng, "[MASK]", max_options=28)
            state = STATES[n % len(STATES)] if n % 3 else req["state"]
            for q in parse_questions(req["questions"]):
                try:
                    aq = native.question(q)
                except ValueError:
                    with pytest.raises(RequestError):
                        builder.ask(q)
                    refused += 1
                    continue
                asked = builder.ask(q)
                labels, ids = resolve_labels(hf, aq)
                perms = anyjev_perms(aq, canonical)
                assert asked.perms == [from_anyjev(q.type, p) for p in perms]
                for s, perm in enumerate(perms):
                    spec = build_prompt(anyjev_state(state), aq, perm, DEFAULT_SYSTEM, labels)
                    text = builder.text(builder.state(state), asked.texts[s])
                    assert text == render_chat(hf, spec)
                    assert asked.reads[s] == label_ids_for_perm(aq, ids, perm)
                    compared += 1
    assert compared > 1500 and refused > 20


@pytest.mark.parametrize(
    ("prior", "combine", "canonical"),
    [
        ("none", "logmean", False),
        ("none", "mean", False),
        ("none", "logmean", True),
        ("content-free", "logmean", False),
        ("content-free", "mean", True),
    ],
)
def test_decisions_are_anyjev_decisions(
    tokens: Tokens, hf: ChatTokenizer, prior: str, combine: str, canonical: bool
) -> None:
    data = package_prompt(tokens, hf, prior=prior, canonical=canonical)
    data["rotations"]["combine"] = combine
    rt = runtime(tokens, data)
    decider = Decider(
        FakeBackend(tokens),
        level="L0",
        prior="content_free" if prior == "content-free" else "none",
        combine=combine,
        shared_prefix=False,
        adaptive_shifts=False,
        canonical_order=canonical,
    )
    rng = random.Random(21)
    worst, answered = 0.0, 0
    for n in range(60):
        req = request(rng, "[MASK]", max_options=26)
        state = STATES[n % len(STATES)] if n % 2 else req["state"]
        parsed = parse_questions(req["questions"])
        try:
            asked = [native.question(q) for q in parsed]
        except ValueError:
            with pytest.raises(RequestError):
                rt.predict(state, req["questions"])
            continue
        ours = rt.predict(state, req["questions"], decimals=None)["answers"]
        theirs = decider.decide(state, asked)
        for q in parsed:
            p = theirs[q.id].probs
            expected = [p[1], p[0]] if q.type == "noul" else list(p)
            if q.type == "noul":
                got = [1.0 - ours[q.id]["noul"], ours[q.id]["noul"]]
            else:
                got = [ours[q.id]["probabilities"][k] for k in q.keys]
            worst = max(worst, max(abs(a - b) for a, b in zip(got, expected, strict=True)))
            answered += 1
    assert answered > 60
    assert worst < 1e-12


def test_the_native_adapter_answers_through_anyjev(tokens: Tokens, hf: ChatTokenizer) -> None:
    rt = runtime(tokens, package_prompt(tokens, hf))
    model = native.AnyJev(FakeBackend(tokens), {"prior": "none"})
    req = {
        "state": "the customer was charged twice",
        "questions": {
            "intent": {
                "type": "choice",
                "instructions": "What does the customer want?",
                "criteria": {"refund": "money back", "cancel": None, "other": None},
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent?",
                "criteria": ["low", "high"],
            },
            "angry": {
                "type": "noul",
                "instructions": "Is the customer angry?",
                "criteria": {"true": "shouting"},
            },
        },
    }
    theirs = model.system_one(req["state"], req["questions"])["answers"]
    ours = rt.predict(req["state"], req["questions"], decimals=None)["answers"]
    assert theirs["intent"]["choice"] == ours["intent"]["choice"]
    for key, p in theirs["intent"]["probabilities"].items():
        assert p == pytest.approx(ours["intent"]["probabilities"][key], abs=1e-12)
    assert theirs["urgency"]["score"] == pytest.approx(ours["urgency"]["score"], abs=1e-12)
    assert theirs["angry"]["noul"] == pytest.approx(ours["angry"]["noul"], abs=1e-12)


def test_the_rotation_math_is_anyjevs() -> None:
    rng = np.random.default_rng(3)
    for k in (2, 3, 7, 26):
        p = rng.dirichlet(np.ones(k), size=k)
        p[0, 0] = 1e-14  # below the clipping
        perms = cyclic_shifts(k)
        assert rotations.cyclic_shifts(list(range(k))) == perms
        for how in rotations.COMBINE:
            np.testing.assert_array_equal(
                rotations.combine(p, perms, how), marginalize(p, perms, how)
            )
        probes = rng.dirichlet(np.ones(k), size=(k, 3))
        prior = rotations.prior_from_probes(probes)
        np.testing.assert_array_equal(prior, content_free_prior(probes))
        for strength in (1.0, 0.75, 0.0):
            np.testing.assert_array_equal(
                rotations.divide_prior(p, prior, strength),
                apply_contextual(p, np.power(prior, strength)),
            )


def test_the_chat_template_is_read_back(hf: ChatTokenizer) -> None:
    template = exporter.chat_template(hf, DEFAULT_SYSTEM)
    assert template == (
        "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )

    class Escaping(ChatTokenizer):
        def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kw):  # type: ignore[no-untyped-def]
            messages = [{**m, "content": m["content"].replace('"', "&quot;")} for m in messages]
            return super().apply_chat_template(messages, tokenize, add_generation_prompt, **kw)

    with pytest.raises(ValueError, match="keep the messages"):
        exporter.chat_template(Escaping(hf.t), DEFAULT_SYSTEM)


def test_the_package_prompt_is_valid(tokens: Tokens, hf: ChatTokenizer) -> None:
    for prior in ("none", "content-free"):
        data = package_prompt(tokens, hf, prior=prior)
        assert schemas.errors(data, "prompt") == []
        Prompt(data)
    assert schemas.errors(exporter.calibration(), "calibration") == []


def broken(data: dict[str, Any], change: Any) -> dict[str, Any]:
    out = copy.deepcopy(data)
    change(out)
    return out


def test_typed_prompts_are_checked(tokens: Tokens, hf: ChatTokenizer) -> None:
    good = package_prompt(tokens, hf)
    choice = lambda d: d["types"]["choice"]  # noqa: E731
    cases = {
        "odxp/0.2": lambda d: d.update(standard="odxp/0.1"),
        "has no question": lambda d: d.update(
            question={"head": "", "option": "{label}{text}", "tail": ""}
        ),
        "has no context.max_tokens": lambda d: d["context"].update(max_tokens=10),
        "choice, score and noul": lambda d: d["types"].pop("score"),
        "one {label}": lambda d: choice(d)["layouts"][0].update(option="{text}"),
        "only {instructions}": lambda d: choice(d)["layouts"][0].update(head="{state}"),
        "a label for each": lambda d: choice(d)["layouts"][0].update(labels=["A", "B"]),
        "distinct": lambda d: choice(d)["layouts"][0].update(labels=["A"] * 26),
        "rotate": lambda d: choice(d).update(rotate="random"),
        "by key": lambda d: choice(d).update(listing=["false", "true"]),
        "exactly one {user}": lambda d: d["chat"].update(template="{system}"),
        "rotations.combine": lambda d: d["rotations"].update(combine="max"),
        "its probes": lambda d: d["rotations"].update(
            prior={"kind": "content-free", "strength": 1}
        ),
        "between 0 and 1": lambda d: d["rotations"].update(
            prior={"kind": "content-free", "probes": ["N/A"], "strength": 2}
        ),
        "state.json": lambda d: d["state"].update(json="yaml"),
    }
    for message, change in cases.items():
        with pytest.raises(PackageError, match=message.replace("{", r"\{").replace("}", r"\}")):
            Prompt(broken(good, change))
    one_type_wrong = broken(good, lambda d: d["types"]["noul"].update(listing=["true", "maybe"]))
    with pytest.raises(PackageError):
        Prompt(one_type_wrong)
    legacy = broken(good, lambda d: d.pop("types"))
    assert schemas.errors(legacy, "prompt")
    with pytest.raises(PackageError, match="need typed layouts"):
        Prompt(
            {
                **legacy,
                "question": {"head": "", "option": "{label} {text}", "tail": ""},
                "labels": ["A", "B"],
                "layouts": [{"max_options": 2, "encoding": "joined"}],
            }
        )


def test_0_1_prompts_cannot_use_0_2_state_options() -> None:
    from opendxp.export import decider as decider_export

    data = decider_export.prompt({}, list("ABCDEFGHIJ"))
    Prompt(data)
    data["state"] = {"json": "indent-2"}
    with pytest.raises(PackageError, match=r"need odxp/0\.2"):
        Prompt(data)
    data["standard"] = "odxp/0.2"
    assert Prompt(data).typed is None


def test_noul_descriptions_follow_the_instructions(tokens: Tokens, hf: ChatTokenizer) -> None:
    builder = TypedBuilder(Prompt(package_prompt(tokens, hf)), tokens)
    [q] = parse_questions(
        {"x": {"type": "noul", "instructions": "Is it late?", "criteria": {"true": "after 9"}}}
    )
    asked = builder.ask(q)
    assert asked.texts == [
        "Question: Is it late?\nYes: after 9\nAnswer Yes or No.",
        "Question: Is it late?\nYes: after 9\nAnswer No or Yes.",
    ]
    assert asked.perms == [[1, 0], [0, 1]]
    yes, no = builder.ids[("noul", 0)][1], builder.ids[("noul", 0)][0]
    assert asked.reads == [[yes, no], [no, yes]]


def test_input_tokens_count_shared_starts_once() -> None:
    assert trie_tokens([]) == 0
    assert trie_tokens([[1, 2, 3], [1, 2, 4], [1, 2, 3]]) == 4
    assert trie_tokens([[5], [1, 2], [1]]) == 3


def test_the_request_set_maps_to_anyjev() -> None:
    """Every question of the 0.1 request set either maps to AnyJev or is refused for a reason."""
    path = Path(__file__).resolve().parents[1] / "src" / "opendxp" / "data" / "requests-0.1.jsonl"
    refused = []
    for line in path.read_text().splitlines():
        case = json.loads(line)
        for q in parse_questions(case["request"]["questions"]):
            try:
                native.question(q)
            except ValueError as exc:
                refused.append((case["id"], str(exc)))
    assert refused == [("r51", "question 'fault' needs instructions")]


def test_unknown_words_still_tokenise(tokens: Tokens) -> None:
    assert tokens.encode(" ".join(UNKNOWN)) == [tokens.id("[UNK]")] * len(UNKNOWN)
    assert len(tokens.encode("10")) == 2
