"""AnyJev (Nokia): its own L0 readout, for writing conformance files.

anyjev 0.2.0's Decider at level L0 on its transformers backend, on the CPU in
float32, one prompt per forward: anyjev.readout builds and renders every
prompt, the backend reads the label logits at the last position, and
anyjev.calibrate combines the rotations. Every rotation is read (adaptive
shifts are a serving choice, SPEC.md 8) and no running prior is kept: the
package's prompt.json names the prior (none or content-free) and the listing
the rotations turn, and the Decider is set up the same way.

What AnyJev has no word for is the System One request. It is mapped here on
its own, not through the package's templates, the way opendxp.export.anyjev
declares it:

    choice  options "name" or "name: description", in the request's order
    score   the level descriptions, lowest first, as ordered levels
    noul    the instructions, then "Yes: ..." and "No: ..." lines for the
            descriptions given, as the question
    every question needs instructions
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from opendxp.request import Question as Asked
from opendxp.request import parse_questions


def question(q: Asked) -> Any:
    """One System One question as an anyjev Question; ValueError if AnyJev can't take it."""
    from anyjev import Question

    if not q.instructions:
        raise ValueError(f"question {q.id!r} needs instructions")
    if q.type == "choice":
        texts = [
            o.name if o.description is None else f"{o.name}: {o.description}" for o in q.options
        ]
        return Question.choice(q.instructions, texts, name=q.id)
    if q.type == "score":
        if any(o.description is None for o in q.options):
            raise ValueError(f"question {q.id!r}: every level needs a description")
        return Question.score(
            q.instructions, levels=[str(o.description) for o in q.options], name=q.id
        )
    false, true = q.options
    legend = "".join(
        f"\n{word}: {o.description}"
        for word, o in (("Yes", true), ("No", false))
        if o.description is not None
    )
    return Question.noul(q.instructions + legend, name=q.id)


def answer(q: Asked, probs: Any) -> dict[str, Any]:
    """A decision in the System One format, unrounded; noul is anyjev's P(Yes)."""
    p = [float(x) for x in probs]
    if q.type == "noul":
        return {"type": "noul", "noul": p[0]}
    keys = q.keys
    best = max(range(len(p)), key=p.__getitem__)
    out: dict[str, Any] = {"type": q.type, "probabilities": dict(zip(keys, p, strict=True))}
    if q.type == "choice":
        out["choice"] = keys[best]
    else:
        out["score"] = sum(i * x for i, x in enumerate(p))
    return out


def settings(package: Path | None) -> dict[str, Any]:
    """The Decider settings a package declares: its prior, listing and system prompt."""
    if package is None:
        return {}
    manifest = json.loads((Path(package) / "odxp.json").read_text(encoding="utf-8"))
    prompt = json.loads((Path(package) / manifest["prompt"]["path"]).read_text(encoding="utf-8"))
    kind = (prompt.get("rotations") or {}).get("prior", {}).get("kind", "none")
    out: dict[str, Any] = {
        "prior": "content_free" if kind == "content-free" else "none",
        "canonical_order": prompt["types"]["choice"].get("listing") == "canonical",
    }
    if "chat" in prompt:
        out["system"] = prompt["chat"].get("system", "")
    return out


class AnyJev:
    """anyjev's Decider at L0 on a backend: the transformers one for conformance files."""

    def __init__(self, backend: Any, settings: dict[str, Any]) -> None:
        from anyjev import Decider

        self.backend, self.settings = backend, settings
        self.decider = Decider(
            backend,
            level="L0",
            combine="logmean",
            shared_prefix=False,
            adaptive_shifts=False,
            **{"prior": "none", **settings},
        )

    @classmethod
    def load(cls, checkpoint: Path, *, threads: int = 4, package: Path | None = None) -> AnyJev:
        import torch
        from anyjev.backends.hf import HFBackend

        torch.set_num_threads(threads)
        backend = HFBackend(str(checkpoint), device="cpu", dtype="float32", batch_size=1)
        return cls(backend, settings(package))

    def system_one(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        parsed = parse_questions(questions)
        asked = [question(q) for q in parsed]
        decisions = self.decider.decide(state, asked)
        return {"answers": {q.id: answer(q, decisions[q.id].probs) for q in parsed}}
