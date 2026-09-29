"""Typed layouts for the causal-letters profile (SPEC.md 6.7 and 6.8, odxp/0.2).

A 0.1 prompt lays every question type out the same way, with one list of
labels. A typed prompt lays each type out on its own: its labels and wording by
option count, whether a label names a position or stays with its option, and
whether the question is asked once or once per rotation of its options. A row
may be wrapped in the model's chat template, and is encoded whole.

This is how an instruction-tuned model is asked for a typed decision without
being trained for it; AnyJev's L0 (Nokia) is written this way.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from opendxp import rotations
from opendxp.answers import softmax
from opendxp.errors import PackageError, RequestError
from opendxp.profiles.encoder_markers import render_option
from opendxp.request import NOUL_KEYS, Question
from opendxp.spec import QUESTION_TYPES
from opendxp.text import fill, placeholders, render_state
from opendxp.tokens import Tokens

LABELS = ("positional", "attached")
ROTATE = ("none", "cyclic")
LISTINGS = ("request", "canonical")
PRIORS = ("none", "content-free")
HEAD_FIELDS = {"instructions", "legend"}
OPTION_FIELDS = {"label", "text"}

# The logits of the given label tokens at the last position of a row of token ids.
Logits = Callable[[list[int], list[int]], np.ndarray]


@dataclass(frozen=True)
class Layout:
    max_options: int
    labels: tuple[str, ...]
    head: str
    option: str
    separator: str
    tail: str


@dataclass(frozen=True)
class TypeLayout:
    """How one question type is asked."""

    type: str
    layouts: tuple[Layout, ...]
    labels: str
    rotate: str
    listing: str | tuple[str, ...]

    @property
    def max_options(self) -> int:
        return self.layouts[-1].max_options

    def layout(self, count: int) -> int:
        """The index of the first layout whose `max_options` holds `count` options."""
        for i, layout in enumerate(self.layouts):
            if count <= layout.max_options:
                return i
        raise RequestError(
            f"this model reads at most {self.max_options} options in a {self.type} question"
        )

    def base(self, question: Question, texts: Sequence[str]) -> list[int]:
        """The first listing: option indices in the order the options are shown."""
        if isinstance(self.listing, tuple):
            return [question.keys.index(key) for key in self.listing]
        return rotations.listing(self.listing, texts)


def _problem(message: str) -> PackageError:
    return PackageError(f"prompt.json: {message}")


def _layout(kind: str, raw: Any) -> Layout:
    fields = ("max_options", "labels", "head", "option", "separator", "tail")
    if not isinstance(raw, dict) or any(key not in raw for key in fields):
        raise _problem(f"a {kind} layout needs {', '.join(fields)}")
    texts = [raw[key] for key in fields[2:]]
    if not all(isinstance(x, str) for x in texts):
        raise _problem(f"the head, option, separator and tail of a {kind} layout are texts")
    if not isinstance(raw["max_options"], int) or not isinstance(raw["labels"], list):
        raise _problem(f"a {kind} layout needs an integer max_options and a list of labels")
    layout = Layout(raw["max_options"], tuple(raw["labels"]), *texts)
    labels = layout.labels
    if not all(isinstance(x, str) and x for x in labels) or len(set(labels)) != len(labels):
        raise _problem(f"the labels of a {kind} layout must be distinct non-empty strings")
    if layout.max_options < 2 or len(labels) < layout.max_options:
        raise _problem(f"a {kind} layout needs a label for each of at least 2 options")
    if layout.option.count("{label}") != 1 or not placeholders(layout.option) <= OPTION_FIELDS:
        raise _problem(f"a {kind} option needs one {{label}} and may have a {{text}}")
    if not placeholders(layout.head) <= HEAD_FIELDS:
        raise _problem(f"a {kind} head may use only {{instructions}} and {{legend}}")
    return layout


def _type(kind: str, raw: Any) -> TypeLayout:
    if not isinstance(raw, dict):
        raise _problem(f"types.{kind} must be an object")
    found = [_layout(kind, x) for x in raw.get("layouts") or []]
    layouts = tuple(sorted(found, key=lambda x: x.max_options))
    if not layouts:
        raise _problem(f"types.{kind} needs at least one layout")
    if len({x.max_options for x in layouts}) != len(layouts):
        raise _problem(f"two {kind} layouts have the same max_options")
    labels = raw.get("labels", "positional")
    rotate = raw.get("rotate", "none")
    listing = raw.get("listing", "request")
    if labels not in LABELS:
        raise _problem(f"types.{kind}.labels must be one of {', '.join(LABELS)}")
    if rotate not in ROTATE:
        raise _problem(f"types.{kind}.rotate must be one of {', '.join(ROTATE)}")
    if isinstance(listing, list):
        if kind != "noul" or sorted(listing) != sorted(NOUL_KEYS):
            raise _problem('only noul lists its options by key, and then as "false" and "true"')
        listing = tuple(listing)
    elif listing not in LISTINGS:
        raise _problem(f"types.{kind}.listing must be one of {', '.join(LISTINGS)} or a key list")
    return TypeLayout(kind, layouts, labels, rotate, listing)


class TypedLayouts:
    """The typed part of a 0.2 prompt.json: `types`, `chat` and `rotations`, checked."""

    def __init__(self, data: dict[str, Any]) -> None:
        types = data.get("types")
        if not isinstance(types, dict) or set(types) != set(QUESTION_TYPES):
            raise _problem("types must lay out choice, score and noul")
        self.types = {kind: _type(kind, types[kind]) for kind in QUESTION_TYPES}
        if self.types["noul"].max_options < 2:
            raise _problem("noul needs a layout for its 2 options")

        chat = data.get("chat")
        self.chat: str | None = None
        self.system = ""
        if chat is not None:
            template = chat.get("template") if isinstance(chat, dict) else None
            if not isinstance(template, str) or template.count("{user}") != 1:
                raise _problem("chat.template needs exactly one {user}")
            if template.count("{system}") > 1:
                raise _problem("chat.template may hold {system} once")
            self.chat, self.system = template, str(chat.get("system", ""))

        rot = data.get("rotations") or {}
        self.combine = rot.get("combine", "logmean")
        if self.combine not in rotations.COMBINE:
            raise _problem(f"rotations.combine must be one of {', '.join(rotations.COMBINE)}")
        prior = rot.get("prior") or {"kind": "none"}
        self.prior = prior.get("kind")
        if self.prior not in PRIORS:
            raise _problem(f"rotations.prior.kind must be one of {', '.join(PRIORS)}")
        self.probes: tuple[str, ...] = ()
        self.strength = 0.0
        if self.prior == "content-free":
            probes = prior.get("probes")
            strength = prior.get("strength")
            if not isinstance(probes, list) or not probes:
                raise _problem("a content-free prior needs its probes, a list of strings")
            if not all(isinstance(x, str) for x in probes):
                raise _problem("a content-free prior needs its probes, a list of strings")
            if isinstance(strength, bool) or not isinstance(strength, (int, float)):
                raise _problem("a content-free prior needs a strength between 0 and 1")
            if not 0.0 <= float(strength) <= 1.0:
                raise _problem("a content-free prior needs a strength between 0 and 1")
            self.probes, self.strength = tuple(probes), float(strength)


@dataclass(frozen=True)
class Asked:
    """One question as a typed prompt asks it: per rotation, the listing, the
    question text and the label tokens read, in position order."""

    perms: list[list[int]]
    texts: list[str]
    reads: list[list[int]]


def trie_tokens(rows: Sequence[Sequence[int]]) -> int:
    """Input tokens counted as a prefix-sharing engine reads them: every shared start once."""
    total, previous = 0, ()
    for row in sorted({tuple(r) for r in rows}):
        common = 0
        while common < min(len(row), len(previous)) and row[common] == previous[common]:
            common += 1
        total += len(row) - common
        previous = row
    return total


class TypedBuilder:
    """Rows and distributions for a typed prompt (SPEC.md 6.7 and 6.8)."""

    PROBE_CACHE = 4096

    def __init__(self, prompt: Any, tokens: Tokens) -> None:
        self.prompt, self.tokens = prompt, tokens
        self.layouts: TypedLayouts = prompt.typed
        self.ids: dict[tuple[str, int], list[int]] = {}
        for kind, spec in self.layouts.types.items():
            for i, layout in enumerate(spec.layouts):
                ids = []
                for label in layout.labels:
                    single = tokens.single(label)
                    if single is None:
                        raise PackageError(f"label {label!r} is not one token of this tokenizer")
                    ids.append(single)
                if len(set(ids)) != len(ids):
                    raise PackageError(f"two {kind} labels encode to the same token")
                self.ids[(kind, i)] = ids
        self.label_ids = sorted({t for ids in self.ids.values() for t in ids})
        self.probe_priors: dict[tuple[str, tuple[int, ...], float], np.ndarray] = {}

    def ask(self, question: Question) -> Asked:
        p, spec = self.prompt, self.layouts.types[question.type]
        which = spec.layout(len(question.options))
        layout, ids = spec.layouts[which], self.ids[(question.type, which)]
        texts = [render_option(p.data["options"], question, o) for o in question.options]
        if question.type == "choice" and len(set(texts)) != len(texts):
            raise RequestError(f"question {question.id!r}: two options read the same")
        base = spec.base(question, texts)
        head = fill(
            layout.head,
            instructions=p.instructions(question),
            legend="".join(texts[i] for i in base),
        )
        positional = spec.labels == "positional"
        perms = rotations.cyclic_shifts(base) if spec.rotate == "cyclic" else [base]
        out = Asked([], [], [])
        for perm in perms:
            shown = [j if positional else i for j, i in enumerate(perm)]
            lines = (
                fill(layout.option, label=layout.labels[k], text=texts[i])
                for k, i in zip(shown, perm, strict=True)
            )
            out.perms.append(list(perm))
            out.texts.append(head + layout.separator.join(lines) + layout.tail)
            out.reads.append([ids[k] for k in shown])
        return out

    def text(self, state_text: str, question_text: str) -> str:
        """A row: the context and the question, in the chat template if there is one."""
        user = fill(self.prompt.context, state=state_text) + question_text
        if self.layouts.chat is None:
            return user
        return fill(self.layouts.chat, system=self.layouts.system, user=user)

    def row(self, state_text: str, question_text: str) -> list[int]:
        """A row's token ids: its text encoded whole."""
        return self.tokens.encode(self.text(state_text, question_text))

    def state(self, state: Any) -> str:
        return render_state(state, self.prompt.data.get("state"))

    def prior(self, text: str, read: list[int], logits: Logits, t: float) -> np.ndarray:
        """The content-free prior of one rotation: the mean distribution over the probes.

        It is read at the question's temperature, so a runtime answering at another
        calibration (SPEC.md 7.1) does not reuse it."""
        key = (text, tuple(read), float(t))
        found = self.probe_priors.get(key)
        if found is None:
            probs = [
                softmax(logits(self.row(self.state(probe), text), read).tolist(), t)
                for probe in self.layouts.probes
            ]
            found = rotations.prior_from_probes(np.asarray(probs))
            if len(self.probe_priors) >= self.PROBE_CACHE:
                self.probe_priors.clear()
            self.probe_priors[key] = found
        return found

    def distribution(
        self, state_text: str, asked: Asked, logits: Logits, t: float
    ) -> tuple[list[float], list[list[int]]]:
        """The question's probabilities in option order, and the rows read for it."""
        pairs = list(zip(asked.texts, asked.reads, strict=True))
        rows = [self.row(state_text, text) for text, _ in pairs]
        p_pos = np.asarray(
            [
                softmax(logits(ids, read).tolist(), t)
                for ids, (_, read) in zip(rows, pairs, strict=True)
            ]
        )
        if self.layouts.prior == "content-free":
            prior = np.stack([self.prior(text, read, logits, t) for text, read in pairs])
            p_pos = rotations.divide_prior(p_pos, prior, self.layouts.strength)
        p = rotations.combine(p_pos, asked.perms, self.layouts.combine)
        return [float(x) for x in p], rows
