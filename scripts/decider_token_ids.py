"""Compare Decider's own prompt token ids with a causal-letters package's.

Runs Decider's own prompt builder (opendxp.native.decider, with the model's
Hugging Face tokenizer through transformers.AutoTokenizer) and the package's
prompt.json builder (with `tokenizers` alone) on every row of the request set,
without running the model, and counts rows whose ids differ.

    python scripts/decider_token_ids.py PACKAGE CHECKPOINT
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from opendxp.conformance import DEFAULT_REQUESTS, read_jsonl
from opendxp.native.decider import load_prompter
from opendxp.package import open_package
from opendxp.profiles.causal_letters import Builder, Prompt
from opendxp.request import parse_questions
from opendxp.tokens import Tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package")
    parser.add_argument("checkpoint")
    parser.add_argument("--requests", default=str(DEFAULT_REQUESTS))
    args = parser.parse_args()

    native, _ = load_prompter(Path(args.checkpoint))
    package = open_package(args.package)
    builder = Builder(Prompt(package.read_json("prompt")), Tokens(package.file("tokenizer")))
    rows = mismatches = 0
    for case in read_jsonl(Path(args.requests)):
        request = case["request"]
        plan, _ = native.plan(request["state"], request["questions"], True)
        context = builder.context(request["state"])
        for (_, _, _, native_rows), q in zip(
            plan, parse_questions(request["questions"]), strict=True
        ):
            _, ours = builder.rows(q)
            for (ids, _), (text, options) in zip(native_rows, ours, strict=True):
                rows += 1
                mismatches += ids != context + builder.question_piece(text, options)
    print(
        json.dumps(
            {
                "rows": rows,
                "mismatches": mismatches,
                "labels_equal": builder.label_ids == native.label_ids,
            }
        )
    )


if __name__ == "__main__":
    main()
