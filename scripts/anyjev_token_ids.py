"""Compare AnyJev's own prompt token ids with a typed causal-letters package's.

For every question of the request set and every rotation AnyJev reads, builds
the row twice without running the model: with anyjev's own code
(anyjev.readout.build_prompt, render_chat and resolve_labels, and the model's
Hugging Face tokenizer through transformers.AutoTokenizer), and with the
package's prompt.json and `tokenizers` alone. Counts rows whose token ids, or
whose label tokens, differ.

    python scripts/anyjev_token_ids.py PACKAGE CHECKPOINT
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from anyjev.readout import build_prompt, label_ids_for_perm, render_chat, resolve_labels
from anyjev.state import render_state
from transformers import AutoTokenizer

from noulxp.conformance import DEFAULT_REQUESTS, read_jsonl
from noulxp.native import anyjev as native
from noulxp.native.anyjev import settings
from noulxp.package import open_package
from noulxp.profiles.causal_letters import Prompt
from noulxp.profiles.typed import TypedBuilder
from noulxp.request import parse_questions
from noulxp.tokens import Tokens


def perms(q: Any, canonical: bool) -> list[list[int]]:
    """anyjev.decider.Decider._perms at L0 with every rotation read."""
    from anyjev.calibrate.permute import cyclic_shifts

    if q.ordered:
        return [list(range(q.k))]
    if q.kind == "noul":
        return [[0, 1], [1, 0]]
    shifts = cyclic_shifts(q.k)
    if not canonical:
        return shifts
    canon = sorted(range(q.k), key=lambda i: (str(q.options[i]), i))
    return [[canon[i] for i in perm] for perm in shifts]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package")
    parser.add_argument("checkpoint")
    parser.add_argument("--requests", default=str(DEFAULT_REQUESTS))
    args = parser.parse_args()

    hf = AutoTokenizer.from_pretrained(args.checkpoint)
    package = open_package(args.package)
    builder = TypedBuilder(Prompt(package.read_json("prompt")), Tokens(package.file("tokenizer")))
    setup = settings(Path(args.package))
    rows = mismatches = label_mismatches = refused = 0
    for case in read_jsonl(Path(args.requests)):
        state = case["request"]["state"]
        for q in parse_questions(case["request"]["questions"]):
            try:
                theirs = native.question(q)
            except ValueError:
                refused += 1
                continue
            asked = builder.ask(q)
            labels, ids = resolve_labels(hf, theirs)
            for s, perm in enumerate(perms(theirs, setup["canonical_order"])):
                spec = build_prompt(render_state(state), theirs, perm, setup["system"], labels)
                expected = hf.encode(render_chat(hf, spec), add_special_tokens=False)
                rows += 1
                mismatches += expected != builder.row(builder.state(state), asked.texts[s])
                label_mismatches += asked.reads[s] != label_ids_for_perm(theirs, ids, perm)
    print(
        json.dumps(
            {
                "rows": rows,
                "mismatches": mismatches,
                "label_mismatches": label_mismatches,
                "questions_refused": refused,
            }
        )
    )


if __name__ == "__main__":
    main()
