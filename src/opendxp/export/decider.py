"""`opendxp export decider`: Decider (Mark Marosi) as a causal-letters package.

The official GGUF build is packaged unchanged. prompt.json reproduces the plain
layout of decider-ai 1.6.0 (github.com/Mapika/decider at 23579f7a, Apache-2.0):
prompt_fast.py (context and question pieces tokenised separately; A-J inside
one encoded question piece up to 10 options, one label token per option beyond),
systemone.py (option texts, isolated score levels, the noul fallback question)
and prompt.py (the 255 single-token labels). calibration.json holds the
temperatures of decider_config.json as temperature.py applies them.
"""

from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Any

from opendxp.export.common import entry, manifest, place, write_json, write_manifest
from opendxp.export.encoder import calibration_ok
from opendxp.spec import BASE, QUESTION_TYPES
from opendxp.tokens import Tokens

NARROW = 10
ISOLATED = "{instructions}\nProposed answer: {level}\nDoes the proposed answer fit?"
LEVEL_NUMBER = r"^\s*-?\d+\s*:\s*"
NOUL_WITHOUT_INSTRUCTIONS = "Which answer fits the context?"
MAX_LEVELS = 10
CONTEXT_TOKENS = 40960  # decider.engine_gguf n_ctx: the longest row the official engine reads


def labels(tokens: Tokens, count: int) -> list[str]:
    """decider.prompt.label_table: A..Z, then the two-letter labels that are single tokens."""
    upper = string.ascii_uppercase
    found = []
    for label in list(upper) + [a + b for a in upper for b in upper]:
        if tokens.single(label) is not None:
            found.append(label)
        if len(found) == count:
            break
    if len(found) != count:
        raise ValueError(f"the tokenizer has only {len(found)} single-token labels")
    return found


def prompt(cfg: dict[str, Any], label_list: list[str]) -> dict[str, Any]:
    return {
        "standard": BASE,
        "profile": "causal-letters",
        "context": {
            "template": "Context:\n{state}",
            "max_tokens": int(cfg.get("max_state_tokens", 32768)),
        },
        "question": {
            "head": "\n\nQuestion: {instructions}\nOptions:",
            "option": "\n({label}) {text}",
            "tail": "\nAnswer: (",
        },
        "labels": label_list,
        "layouts": [
            {"max_options": NARROW, "encoding": "joined"},
            {"max_options": len(label_list), "encoding": "split-at-label"},
        ],
        "slot": "last",
        "options": {
            "choice": {"described": "{name}: {description}", "bare": "{name}"},
            "score": {"described": "{index}: {description}"},
            "noul": {
                "false": {"described": "no: {description}", "bare": "no"},
                "true": {"described": "yes: {description}", "bare": "yes"},
            },
        },
        "score": (
            {
                "mode": "isolated",
                "question": ISOLATED,
                "level_strip": LEVEL_NUMBER,
                "options": ["no", "yes"],
                "read": 1,
            }
            if cfg.get("isolated_levels")
            else {"mode": "list"}
        ),
        "instructions": {"fallback": {"noul": NOUL_WITHOUT_INSTRUCTIONS}, "required": True},
        "state": {"index_arrays_from": 8},
        # decider.engine_gguf: llama.cpp's default attention and cache, 2048-token micro-batches.
        "decode": {"flash_attention": "auto", "kv_cache": "f16", "ubatch": 2048},
    }


def calibration(cfg: dict[str, Any]) -> dict[str, Any]:
    default = float(cfg.get("temperature", 1.0))
    by_type = cfg.get("temperature_by_type") or {}
    return calibration_ok(
        {
            "standard": BASE,
            "temperature": {t: float(by_type.get(t, default)) for t in QUESTION_TYPES},
            "source": {
                "file": "decider_config.json",
                "temperature": cfg.get("temperature"),
                "temperature_by_type": by_type,
                "rule": "decider.temperature: temperature_by_type[type], else temperature; "
                "every level row of an isolated score uses the score temperature",
            },
        }
    )


def export(
    checkpoint: Path,
    out_dir: Path,
    *,
    name: str = "mapika/decider",
    weights_mode: str = "link",
    gguf: str | None = None,
    source: dict[str, Any] | None = None,
    log: Any = print,
) -> dict[str, Any]:
    checkpoint, out_dir = Path(checkpoint), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((checkpoint / "decider_config.json").read_text())
    layout = cfg.get("layout") or ("chat" if cfg.get("chat_template") is True else "plain")
    if layout != "plain":
        raise ValueError(f"only Decider's plain layout maps to causal-letters, not {layout!r}")
    weights = checkpoint / gguf if gguf else next(iter(sorted(checkpoint.glob("*.gguf"))), None)
    if weights is None or not weights.exists():
        raise FileNotFoundError(f"no .gguf file in {checkpoint}")
    with open(weights, "rb") as fh:
        if fh.read(4) != b"GGUF":
            raise ValueError(f"{weights.name} is not a GGUF file")
    how = {
        "weights": place(weights, out_dir / "model.gguf", weights_mode),
        "tokenizer": place(checkpoint / "tokenizer.json", out_dir / "tokenizer.json", weights_mode),
    }
    log(f"placed weights ({how['weights']}) and tokenizer ({how['tokenizer']})")
    tokens = Tokens(out_dir / "tokenizer.json")
    label_list = labels(tokens, int(cfg.get("max_options", 255)))
    if label_list[:NARROW] != list("ABCDEFGHIJ"):
        raise ValueError("the tokenizer's letters A-J are not single tokens")
    write_json(out_dir / "prompt.json", prompt(cfg, label_list))
    write_json(out_dir / "calibration.json", calibration(cfg))
    files = {
        "weights": entry(out_dir, "model.gguf", format="gguf"),
        "tokenizer": entry(out_dir, "tokenizer.json"),
        "prompt": entry(out_dir, "prompt.json"),
        "calibration": entry(out_dir, "calibration.json"),
    }
    data = manifest(
        name=name,
        profile="causal-letters",
        files=files,
        limits={
            "max_tokens": CONTEXT_TOKENS,
            "max_options": len(label_list),
            "max_levels": MAX_LEVELS,
        },
        confidence={"choice": "typesafe", "score": "typesafe-ordinal"},
        source={
            "model": "Mapika/decider-2b-GGUF",
            "version": cfg.get("version"),
            "license": "Apache-2.0",
            # The folder's name, not where it sits: packages get published.
            "checkpoint": Path(checkpoint).name,
            "weights_file": weights.name,
            **(source or {}),
        },
    )
    write_manifest(out_dir, data)
    return data
