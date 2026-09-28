"""Decider (Mark Marosi): its own inference, for writing conformance files.

A Qwen3.5 language model that reads answer-letter logits. For every question
the prompt is the state, the question and its lettered options, ending at an
answer slot; the answer is the softmax, at the declared temperature, of the
logits of the option letters at that slot. Score questions ask one yes/no row
per level ("isolated levels") and normalise.

Rebuilt from github.com/Mapika/decider at 23579f7a (decider-ai 1.6.0,
Apache-2.0): systemone.py (rendering, rows, answer format), prompt.py and
prompt_fast.py (the plain layout and label ids), temperature.py, and
engine_gguf.py / decide_gguf.py (the llama.cpp readout). As in engine_gguf.py,
every row is decoded on its own, from empty memory, on the CPU, with token ids
from the model's own Hugging Face tokenizer.
"""

from __future__ import annotations

import ctypes
import json
import math
import re
import string
from pathlib import Path
from typing import Any

from opendxp.errors import PackageError
from opendxp.profiles.causal_letters import _silence

MAX_CHOICE, MAX_LEVELS, ANNOTATE_MIN = 255, 10, 8
NARROW, MAX_OPTIONS, LETTERS = 10, 255, "ABCDEFGHIJ"
N_CTX = 8192
NOUL_WITHOUT_INSTRUCTIONS = "Which answer fits the context?"
ISOLATED = "{q}\nProposed answer: {level}\nDoes the proposed answer fit?"


def _txt(v: Any) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def _annotate(x: Any) -> Any:
    if isinstance(x, list):
        if len(x) >= ANNOTATE_MIN:
            return [
                {"_index": i, **_annotate(v)}
                if isinstance(v, dict)
                else {"_index": i, "value": _annotate(v)}
                for i, v in enumerate(x)
            ]
        return [_annotate(v) for v in x]
    if isinstance(x, dict):
        return {k: _annotate(v) for k, v in x.items()}
    return x


def _render_state(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(_annotate(state), ensure_ascii=False)


def _render_question(spec: dict[str, Any]) -> dict[str, Any]:
    """systemone.py:40-74; a ValueError here is a question the model can't take."""
    t = spec.get("type", "choice")
    crit = spec.get("criteria", spec.get("options"))
    raw = spec.get("instructions", spec.get("question", ""))
    if t in ("noul", "bool") and raw in (None, ""):
        ins = NOUL_WITHOUT_INSTRUCTIONS
    else:
        ins = _txt(raw)
    if not ins:
        raise ValueError("question without instructions")
    names: list[Any]
    if t == "choice":
        if isinstance(crit, (list, tuple)):
            crit = {str(c): None for c in crit}
        if not isinstance(crit, dict) or not 2 <= len(crit) <= MAX_CHOICE:
            raise ValueError(f"choice criteria: a map of 2..{MAX_CHOICE} options")
        names = list(crit)
        opts = [n if crit[n] in (None, "") else f"{n}: {_txt(crit[n])}" for n in names]
    elif t == "score":
        if isinstance(crit, dict):
            crit = [crit[k] for k in sorted(crit, key=float)]
        if not isinstance(crit, (list, tuple)) or not 2 <= len(crit) <= MAX_LEVELS:
            raise ValueError(f"score criteria: an ordered list of 2..{MAX_LEVELS} levels")
        names = list(range(len(crit)))
        opts = [f"{i}: {_txt(c)}" for i, c in enumerate(crit)]
    elif t in ("noul", "bool"):
        if crit is not None and not isinstance(crit, dict):
            raise ValueError("noul criteria: a map of optional true/false descriptions")
        names = [False, True]
        c = crit if crit is not None else {}
        f, tr = c.get("false"), c.get("true")
        opts = [
            "no" if f in (None, "") else f"no: {_txt(f)}",
            "yes" if tr in (None, "") else f"yes: {_txt(tr)}",
        ]
    else:
        raise ValueError(f"unknown question type {t!r}")
    return {
        "question": ins,
        "options": opts,
        "type": "noul" if t == "bool" else t,
        "names": names,
        "legend": [_txt(c) for c in crit] if t == "score" and crit is not None else None,
    }


def _rows(rq: dict[str, Any], isolated_levels: bool) -> tuple[str, list[tuple[str, list[str]]]]:
    if isolated_levels and rq["type"] == "score":
        return "iso", [
            (
                ISOLATED.format(q=rq["question"], level=re.sub(r"^\s*-?\d+\s*:\s*", "", level)),
                ["no", "yes"],
            )
            for level in rq["legend"]
        ]
    return "list", [(rq["question"], list(rq["options"]))]


def _norm(p: list[float]) -> list[float]:
    total = sum(p)
    return [1.0 / len(p)] * len(p) if total == 0 else [x / total for x in p]


def _certainty(p: list[float]) -> float:
    h = -sum(x * math.log(x) for x in p if x > 0)
    return max(0.0, 1.0 - h / math.log(len(p))) if len(p) > 1 else 1.0


def _choice_confidence(p: list[float]) -> float:
    n = len(p)
    p = _norm(p)
    return 1.0 if n <= 1 else min(1.0, max(0.0, (n * max(p) - 1) / (n - 1)))


def _score_confidence(p: list[float]) -> float:
    n = len(p)
    p = _norm(p)
    if n <= 1:
        return 1.0
    k = max(range(n), key=p.__getitem__)
    spread = sum(x * abs(i - k) for i, x in enumerate(p))
    uniform = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return min(1.0, max(0.0, 1.0 - spread / uniform))


def _format(rq: dict[str, Any], p: list[float], nd: int = 4) -> dict[str, Any]:
    """systemone.py:176-189."""
    p = [float(x) for x in p[: len(rq["options"])]]
    s = sum(p) or 1.0
    p = [x / s for x in p]
    j = max(range(len(p)), key=p.__getitem__)
    if rq["type"] == "noul":
        return {"type": "noul", "noul": round(p[1], nd)}
    if rq["type"] == "choice":
        return {
            "type": "choice",
            "choice": rq["names"][j],
            "confidence": round(_choice_confidence(p), nd),
            "x_p_max": round(p[j], nd),
            "certainty": round(_certainty(p), nd),
            "probabilities": {n: round(x, nd) for n, x in zip(rq["names"], p, strict=True)},
        }
    return {
        "type": "score",
        "score": round(sum(i * x for i, x in enumerate(p)), 2),
        "confidence": round(_score_confidence(p), nd),
        "x_p_max": round(p[j], nd),
        "certainty": round(_certainty(p), nd),
        "legend": {str(i): d for i, d in enumerate(rq["legend"])},
        "probabilities": {str(i): round(x, nd) for i, x in enumerate(p)},
    }


def _assemble(rq: dict[str, Any], kind: str, row_probs: list[list[float]]) -> dict[str, Any]:
    if kind == "iso":
        fit = [float(pr[1]) for pr in row_probs]
        total = sum(fit) or 1e-9
        answer = _format(rq, [x / total for x in fit])
        answer["level_fit"] = {str(j): round(x, 4) for j, x in enumerate(fit)}
        answer["fit_mass"] = round(total, 4)
        return answer
    return _format(rq, row_probs[0])


def _temperature(cfg: dict[str, Any], qtype: str) -> float:
    default = float(cfg.get("temperature", 1.0))
    return float((cfg.get("temperature_by_type") or {}).get(qtype, default))


def _unique_tokens(context: int, suffixes: list[list[int]]) -> int:
    if len(suffixes) < 2:
        return context + sum(len(s) for s in suffixes)
    lcp, short, first = 0, min(len(s) for s in suffixes), suffixes[0]
    while lcp < short and all(s[lcp] == first[lcp] for s in suffixes):
        lcp += 1
    return context + lcp + sum(len(s) - lcp for s in suffixes)


class Prompter:
    """Token ids exactly as decider.prompt and prompt_fast build them: the plain layout."""

    def __init__(self, tok: Any) -> None:
        self.tok = tok
        upper = string.ascii_uppercase
        ids: list[int] = []
        for label in list(upper) + [a + b for a in upper for b in upper]:
            t = tok.encode(label, add_special_tokens=False)
            if len(t) == 1:
                ids.append(t[0])
            if len(ids) == MAX_OPTIONS:
                break
        if len(ids) != MAX_OPTIONS or len(set(ids)) != MAX_OPTIONS:
            raise PackageError("the tokenizer has no single-token option labels")
        for j, letter in enumerate(LETTERS):
            if tok.encode(letter, add_special_tokens=False) != [ids[j]]:
                raise PackageError("the tokenizer's letters do not match Decider's labels")
        self.label_ids = ids
        self.open_ids = tok.encode("\n(", add_special_tokens=False)

    def enc(self, s: str) -> list[int]:
        ids: list[int] = self.tok.encode(s, add_special_tokens=False)
        return ids

    def question_piece(self, text: str, options: list[str]) -> list[int]:
        head, tail = f"\n\nQuestion: {text}\nOptions:", "\nAnswer: ("
        if len(options) <= NARROW:
            return self.enc(
                head + "".join(f"\n({LETTERS[j]}) {o}" for j, o in enumerate(options)) + tail
            )
        piece = self.enc(head)
        for j, o in enumerate(options):
            piece += self.open_ids + [self.label_ids[j]] + self.enc(f") {o}")
        return piece + self.enc(tail)

    def plan(
        self, state: Any, questions: dict[str, Any], isolated_levels: bool
    ) -> tuple[list[tuple[str, dict[str, Any], str, list[tuple[list[int], int]]]], int]:
        rqs = {k: _render_question(v) for k, v in questions.items()}
        context = self.enc("Context:\n" + _render_state(state))
        plan, suffixes = [], []
        for k, rq in rqs.items():
            kind, rows = _rows(rq, isolated_levels)
            pieces = [self.question_piece(t, o) for t, o in rows]
            plan.append(
                (
                    k,
                    rq,
                    kind,
                    [(context + s, len(o)) for s, (_, o) in zip(pieces, rows, strict=True)],
                )
            )
            suffixes += pieces
        return plan, _unique_tokens(len(context), suffixes)


def load_prompter(checkpoint: Path) -> tuple[Prompter, dict[str, Any]]:
    """The model's own tokenizer and config, for building prompts without the model."""
    from transformers import AutoTokenizer

    cfg = json.loads((checkpoint / "decider_config.json").read_text(encoding="utf-8"))
    return Prompter(AutoTokenizer.from_pretrained(str(checkpoint))), cfg


class Decider:
    """engine_gguf.py's readout on the CPU: every row decoded in full, from empty memory."""

    def __init__(self, checkpoint: Path, threads: int = 4) -> None:
        import llama_cpp as lc
        import numpy as np

        self.lc, self.np = lc, np
        self.prompter, self.cfg = load_prompter(checkpoint)
        self.isolated_levels = bool(self.cfg.get("isolated_levels", False))
        self.letters = np.asarray(self.prompter.label_ids)
        gguf = checkpoint / "model.gguf"
        if not gguf.is_file():
            found = sorted(checkpoint.glob("*.gguf"))
            if not found:
                raise PackageError(f"no .gguf file in {checkpoint}")
            gguf = found[0]
        _silence(lc)
        lc.llama_backend_init()
        params = lc.llama_model_default_params()
        params.n_gpu_layers = 0  # the reference is the CPU
        # llama.cpp handles, freed by close().
        self.batch: Any = None
        self.ctx: Any = None
        self.model: Any = lc.llama_model_load_from_file(str(gguf).encode(), params)
        if not self.model:
            raise PackageError("llama.cpp could not load the GGUF file")
        cp = lc.llama_context_default_params()
        # A whole row goes to one decode call, in 2,048-token micro-batches.
        cp.n_ctx = N_CTX
        cp.n_batch = N_CTX
        cp.n_ubatch = 2048
        cp.n_threads = cp.n_threads_batch = threads
        self.ctx = lc.llama_init_from_model(self.model, cp)
        if not self.ctx:
            raise PackageError("llama.cpp could not create a context")
        self.memory = lc.llama_get_memory(self.ctx)
        self.n_vocab = lc.llama_vocab_n_tokens(lc.llama_model_get_vocab(self.model))
        self.batch = lc.llama_batch_init(N_CTX, 0, 1)

    def _row(self, ids: list[int], n: int, temperature: float) -> list[float]:
        lc, np, b = self.lc, self.np, self.batch
        lc.llama_memory_clear(self.memory, True)
        for i, t in enumerate(ids):
            b.token[i] = t
            b.pos[i] = i
            b.n_seq_id[i] = 1
            b.seq_id[i][0] = 0
            b.logits[i] = i == len(ids) - 1
        b.n_tokens = len(ids)
        if lc.llama_decode(self.ctx, b) != 0:
            raise RuntimeError("llama_decode failed")
        pointer = ctypes.cast(
            lc.llama_get_logits_ith(self.ctx, len(ids) - 1), ctypes.POINTER(ctypes.c_float)
        )
        z = np.ctypeslib.as_array(pointer, shape=(self.n_vocab,))[self.letters[:n]].astype(
            np.float64
        ) / float(temperature)
        e = np.exp(z - z.max())
        return list((e / e.sum()).tolist())

    def system_one(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        plan, used = self.prompter.plan(state, questions, self.isolated_levels)
        rows = [
            (ids, n, _temperature(self.cfg, rq["type"])) for _, rq, _, rs in plan for ids, n in rs
        ]
        longest = max(len(ids) for ids, _, _ in rows)
        if longest > N_CTX:
            raise ValueError(f"the prompt is {longest} tokens; this reference reads up to {N_CTX}")
        probs = iter([self._row(ids, n, t) for ids, n, t in rows])
        answers = {k: _assemble(rq, kind, [next(probs) for _ in rs]) for k, rq, kind, rs in plan}
        return {"answers": answers, "usage": {"input_tokens": used, "output_tokens": 0}}

    def close(self) -> None:
        """llama.cpp's memory is its own: free the batch, the context and the model, once."""
        lc = self.lc
        if getattr(self, "batch", None) is not None:
            lc.llama_batch_free(self.batch)
            self.batch = None
        if getattr(self, "ctx", None):
            lc.llama_free(self.ctx)
            self.ctx = None
        if getattr(self, "model", None):
            lc.llama_model_free(self.model)
            self.model = None
