"""Run a Julia 1 OpenDXP package on the authors' own parity cases.

SupersonicLabs/Julia-1-ONNX ships parity-cases.json: 100 validation rows
with the logits the original PyTorch model gave (max_length 1024,
head_length 256, strict encoding). Under strict encoding the budgets only
decide acceptance, so the package's template gives the same token ids; this
compares the package's logits and argmax with the published ones.

    python scripts/julia_parity.py PACKAGE parity-cases.json [--device cpu]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from opendxp.package import open_package
from opendxp.profiles.encoder_markers import EncoderMarkersRuntime, pack
from opendxp.request import parse_question


def question(row: dict) -> dict:
    """A parity row as a System One question whose option texts are the row's options."""
    kind, options = row["type"], row["options"]
    if kind == "score":
        return {"type": "score", "instructions": row["question"], "criteria": options}
    if kind == "noul":
        criteria = {"false": options[0], "true": options[1]}
        return {"type": "noul", "instructions": row["question"], "criteria": criteria}
    criteria = {f"option {i}": text for i, text in enumerate(options)}
    return {"type": "choice", "instructions": row["question"], "criteria": criteria}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package")
    parser.add_argument("cases")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    runtime = EncoderMarkersRuntime(open_package(args.package), provider=args.device)
    cases = json.loads(Path(args.cases).read_text())
    worst_logit = worst_p = 0.0
    agree = 0
    for case in cases:
        row = case["request"]
        q = parse_question("q", question(row))
        encoded = runtime.encode(row["state"], [q])
        logits = runtime.run(pack(encoded, runtime.pad_id))[0, : len(q.options)].astype(np.float64)
        want = np.asarray(case["pytorch_logits"], dtype=np.float64)
        worst_logit = max(worst_logit, float(np.abs(logits - want).max()))
        p, w = np.exp(logits - logits.max()), np.exp(want - want.max())
        worst_p = max(worst_p, float(np.abs(p / p.sum() - w / w.sum()).max()))
        agree += int(logits.argmax() == want.argmax())
    print(
        json.dumps(
            {
                "cases": len(cases),
                "argmax_agreement": agree,
                "max_abs_logit": worst_logit,
                "max_abs_p": worst_p,
                "device": args.device,
            }
        )
    )


if __name__ == "__main__":
    main()
