"""`opendxp export anyjev`: a causal LM asked the AnyJev way (Nokia), as a causal-letters package.

AnyJev turns an instruction-tuned LM into a decision model without training it
(github.com/nokia-applied-research/AnyJev, anyjev 0.2.0, Apache-2.0). At level
L0 it needs no labels: a choice is asked once per cyclic shift of its options
and a noul once per order of its answers, the label logits are read at the
answer slot, and the shifts are combined by their log-mean.

prompt.json is typed (odxp/0.2) and reproduces anyjev.readout.build_prompt (the
three question layouts, with the labels resolve_labels picks for this
tokenizer), the model's chat template as anyjev.readout.render_chat renders it
(the default system prompt, no thinking), and anyjev.state.render_state. The
Decider's other L0 settings are the ones a package can keep: every shift is
read, the caller's listing is turned (or the canonical one, --canonical-order),
and there is no prior, or the content-free one (--prior content-free). AnyJev's
default batch prior is a running mean over the requests one Decider has seen,
so an answer would depend on the requests before it (SPEC.md 8).

What AnyJev has no word for is the System One request. It is mapped the same
way here and in opendxp.native.anyjev: a choice option reads "name" or
"name: description", a score level is its description, and a noul's
descriptions follow its instructions as "Yes: ..." and "No: ..." lines.

The weights are a GGUF of the model (--gguf). AnyJev runs the model's own
weights in float32; the GGUF that reproduces it is those weights at their own
precision (F16 for BF16 weights, from llama.cpp's convert_hf_to_gguf.py),
decoded with an f32 cache. A quantised GGUF is a derived package (SPEC.md 10):
on Qwen3-1.7B, Qwen's Q8_0 file changes 2 decisions of 90 (VALIDATION.md).
"""

from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Any

from opendxp.export.common import entry, manifest, place, write_json, write_manifest
from opendxp.export.encoder import calibration_ok
from opendxp.spec import QUESTION_TYPES
from opendxp.text import fill
from opendxp.tokens import Tokens

STANDARD = "odxp/0.2"
ANYJEV = "0.2.0"
LETTERS = string.ascii_uppercase
MAX_CHOICE = 26  # anyjev.question.MAX_OPTIONS: the letter readout
MAX_LEVELS = 10  # anyjev.question.Question.score: 2 to 10 levels
PROBES = ["N/A", "", "[MASK]"]  # anyjev.calibrate.contextual.DEFAULT_PROBES
PRIOR_STRENGTH = 1.0  # anyjev.decider.Decider.DEFAULT_PRIOR_STRENGTH["content_free"]

CONTEXT = "State:\n{state}\n\n"
CHOICE_HEAD = "Question: {instructions}\nOptions:\n"
SCORE_HEAD = "Question: {instructions}\nPick the level that applies (the levels are ordered):\n"
NOUL_HEAD = "Question: {instructions}{legend}\nAnswer "
LETTER_TAIL = "\nAnswer with the letter only."
NUMBER_TAIL = "\nAnswer with the number only."
OPTIONS = {
    "choice": {"described": "{name}: {description}", "bare": "{name}"},
    "score": {"described": "{description}"},
    "noul": {
        "false": {"described": "\nNo: {description}", "bare": ""},
        "true": {"described": "\nYes: {description}", "bare": ""},
    },
}
# The chat template is rendered once with these in place of the conversation.
SYSTEM_MARK = "⁣ODXP_SYSTEM⁣"
USER_MARK = "⁣ODXP_USER⁣"


def _single(tokens: Tokens, label: str) -> None:
    """anyjev.readout.map_label_tokens reads the bare label's token when it is one token."""
    if tokens.single(label) is None:
        raise ValueError(f"the label {label!r} is not one token of this tokenizer")


def _listed(max_options: int, labels: list[str], head: str, tail: str) -> dict[str, Any]:
    """anyjev.readout.build_prompt's option list: "A. text" lines, then the tail."""
    return {
        "max_options": max_options,
        "labels": labels,
        "head": head,
        "option": "{label}. {text}",
        "separator": "\n",
        "tail": tail,
    }


def types(tokens: Tokens, canonical: bool = False) -> dict[str, Any]:
    """The three layouts, with the labels anyjev.readout.resolve_labels picks for `tokens`."""
    for label in [*LETTERS, "Yes", "No"]:
        _single(tokens, label)
    # Score: the digits 1..k while every one of them is a single token, else letters.
    digits = 0
    while digits < MAX_LEVELS and tokens.single(str(digits + 1)) is not None:
        digits += 1
    score = []
    if digits >= 2:
        score.append(_listed(digits, [str(i + 1) for i in range(digits)], SCORE_HEAD, NUMBER_TAIL))
    if digits < MAX_LEVELS:
        score.append(_listed(MAX_LEVELS, list(LETTERS[:MAX_LEVELS]), SCORE_HEAD, LETTER_TAIL))
    return {
        "choice": {
            "layouts": [_listed(MAX_CHOICE, list(LETTERS), CHOICE_HEAD, LETTER_TAIL)],
            "labels": "positional",
            "rotate": "cyclic",
            "listing": "canonical" if canonical else "request",
        },
        # Levels are ordered: asked once, in their order.
        "score": {"layouts": score, "labels": "positional", "rotate": "none"},
        # "Answer Yes or No." and "Answer No or Yes.": the labels stay with their answers.
        "noul": {
            "layouts": [
                {
                    "max_options": 2,
                    "labels": ["No", "Yes"],
                    "head": NOUL_HEAD,
                    "option": "{label}",
                    "separator": " or ",
                    "tail": ".",
                }
            ],
            "labels": "attached",
            "rotate": "cyclic",
            "listing": ["true", "false"],
        },
    }


def chat_template(tokenizer: Any, system: str) -> str:
    """The model's chat template with {system} and {user} in place, as anyjev renders it.

    Rendered once with marks for the two messages, then checked: the marks
    appear once each, and real conversations render exactly as the template
    filled with them.
    """
    from anyjev.readout import PromptSpec, render_chat

    text = render_chat(tokenizer, PromptSpec(system=SYSTEM_MARK, user=USER_MARK))
    if text.count(SYSTEM_MARK) != 1 or text.count(USER_MARK) != 1:
        raise ValueError("the chat template changes the messages it is given")
    template = text.replace(SYSTEM_MARK, "{system}").replace(USER_MARK, "{user}")
    samples = [
        "State:\n(empty)\n\nQuestion: Is it late?\nAnswer Yes or No.",
        'State:\n{\n  "order": "#{123}"\n}\n\nQuestion: Which?\nOptions:\nA. x\nB. y: z\n'
        "Answer with the letter only.",
        "State:\nuser: hola  \nassistant: ¿qué tal?\n\nQuestion: Tone?\nOptions:\nA. warm\n"
        "B. cold\nAnswer with the letter only.",
    ]
    for user in samples:
        rendered = render_chat(tokenizer, PromptSpec(system=system, user=user))
        if rendered != fill(template, system=system, user=user):
            raise ValueError("the chat template does not keep the messages as they are")
    return template


def prompt(
    tokens: Tokens,
    template: str,
    system: str,
    *,
    canonical: bool = False,
    prior: str = "none",
) -> dict[str, Any]:
    if prior not in ("none", "content-free"):
        raise ValueError('prior must be "none" or "content-free"')
    return {
        "standard": STANDARD,
        "profile": "causal-letters",
        "chat": {"template": template, "system": system},
        "context": {"template": CONTEXT},
        "types": types(tokens, canonical),
        "rotations": {
            "combine": "logmean",
            "prior": (
                {"kind": "content-free", "probes": PROBES, "strength": PRIOR_STRENGTH}
                if prior == "content-free"
                else {"kind": "none"}
            ),
        },
        "slot": "last",
        "options": OPTIONS,
        "instructions": {"required": True},
        # anyjev.state.render_state: text as it is, a conversation as "role: content"
        # lines, anything else as JSON indented by two; build_prompt: "(empty)".
        "state": {"json": "indent-2", "messages": "role-content", "empty": "(empty)"},
        # AnyJev's own engine is transformers in float32: llama.cpp's closest settings
        # keep the cache in f32 and attention unfused (VALIDATION.md).
        "decode": {"flash_attention": "disabled", "kv_cache": "f32", "ubatch": 2048},
    }


def calibration() -> dict[str, Any]:
    return calibration_ok(
        {
            "standard": STANDARD,
            "temperature": dict.fromkeys(QUESTION_TYPES, 1.0),
            "source": {
                "rule": "AnyJev L0 reads the label logits at temperature 1; its L1 "
                "temperatures are fit per question on labels, not per model",
            },
        }
    )


def export(
    checkpoint: Path,
    out_dir: Path,
    *,
    gguf: str | Path,
    name: str | None = None,
    base: str | None = None,
    canonical: bool = False,
    prior: str = "none",
    weights_mode: str = "link",
    source: dict[str, Any] | None = None,
    log: Any = print,
) -> dict[str, Any]:
    from anyjev.readout import DEFAULT_SYSTEM
    from transformers import AutoTokenizer

    checkpoint, out_dir = Path(checkpoint), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = Path(gguf)
    if not weights.is_absolute() and not weights.exists():
        weights = checkpoint / weights
    with open(weights, "rb") as fh:
        if fh.read(4) != b"GGUF":
            raise ValueError(f"{weights.name} is not a GGUF file")
    if not (checkpoint / "tokenizer.json").is_file():
        raise FileNotFoundError(f"{checkpoint} has no tokenizer.json")
    how = {
        "weights": place(weights, out_dir / "model.gguf", weights_mode),
        "tokenizer": place(checkpoint / "tokenizer.json", out_dir / "tokenizer.json", weights_mode),
    }
    log(f"placed weights ({how['weights']}) and tokenizer ({how['tokenizer']})")
    tokens = Tokens(out_dir / "tokenizer.json")
    hf = AutoTokenizer.from_pretrained(str(checkpoint))
    template = chat_template(hf, DEFAULT_SYSTEM)
    probe = fill(
        template, system=DEFAULT_SYSTEM, user="State:\nx\n\nQuestion: y?\nAnswer Yes or No."
    )
    if tokens.encode(probe) != list(hf.encode(probe, add_special_tokens=False)):
        raise ValueError("tokenizer.json does not encode as the model's own tokenizer does")
    data = prompt(tokens, template, DEFAULT_SYSTEM, canonical=canonical, prior=prior)
    write_json(out_dir / "prompt.json", data)
    write_json(out_dir / "calibration.json", calibration())
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    model = base or checkpoint.name
    files = {
        "weights": entry(out_dir, "model.gguf", format="gguf"),
        "tokenizer": entry(out_dir, "tokenizer.json"),
        "prompt": entry(out_dir, "prompt.json"),
        "calibration": entry(out_dir, "calibration.json"),
    }
    out = manifest(
        name=name or f"nokia/anyjev-{model.split('/')[-1].lower()}",
        profile="causal-letters",
        files=files,
        limits={
            "max_tokens": int(config.get("max_position_embeddings", 32768)),
            "max_options": MAX_CHOICE,
            "max_levels": MAX_LEVELS,
        },
        # anyjev.result.Decision.confidence: the largest probability.
        confidence=dict.fromkeys(QUESTION_TYPES, "max-probability"),
        source={
            "model": model,
            "method": "AnyJev L0 (github.com/nokia-applied-research/AnyJev)",
            "anyjev": ANYJEV,
            "license": "Apache-2.0",
            "checkpoint": checkpoint.name,
            "weights_file": weights.name,
            **(source or {}),
        },
        standard=STANDARD,
    )
    write_manifest(out_dir, out)
    return out
