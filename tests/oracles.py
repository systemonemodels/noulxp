"""The models' own input construction, ported verbatim, as test oracles.

- laya_*: laya 0.3.5, laya/common.py render_criterion, render_options,
  build_sequence and laya/agent.py _to_internal (Apache-2.0).
- julia_*: the System One Engine's Julia runtime _rows and _sequence, itself
  julia/data.py sequence() with strict encoding (SupersonicLabs/Julia-1,
  Apache-2.0), with the token limits as parameters.
- decider_*: the System One Engine's Decider runtime (_render_question, _rows,
  _Prompter), itself decider-ai 1.6.0 systemone.py / prompt_fast.py (Apache-2.0).

Only the tokenizer calls are adapted: `tok` is a small wrapper with the
transformers-style attributes these functions use.
"""

from __future__ import annotations

import json
import re
import string
from typing import Any

QTYPES = {"choice": 0, "score": 1, "noul": 2}


class HFStyle:
    """tokenizer(text, add_special_tokens=False)["input_ids"] and the special-token attributes."""

    def __init__(self, tokens, mask="[MASK]", cls="[CLS]", sep="[SEP]", pad="[PAD]"):  # type: ignore[no-untyped-def]
        self.t = tokens
        self.mask_token = mask
        self.mask_token_id = tokens.id(mask)
        self.cls_token_id = tokens.id(cls)
        self.sep_token_id = tokens.id(sep)
        self.pad_token_id = tokens.id(pad)

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": self.t.encode(text)}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.t.encode(text)


# ---- Laya -------------------------------------------------------------------------------


def laya_render_criterion(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def laya_render_options(q: dict[str, Any]) -> list[str]:
    t, crit = q["t"], q.get("crit")
    if t == "choice":
        return [
            k if v is None or v == "" else "%s: %s" % (k, laya_render_criterion(v))
            for k, v in crit.items()
        ]
    if t == "score":
        return ["level %d: %s" % (i, laya_render_criterion(c)) for i, c in enumerate(crit)]
    crit = crit or {}
    false_crit, true_crit = crit.get("false"), crit.get("true")
    return [
        "false: "
        + (
            laya_render_criterion(false_crit)
            if false_crit not in (None, "")
            else "no, the statement does not hold"
        ),
        "true: "
        + (
            laya_render_criterion(true_crit)
            if true_crit not in (None, "")
            else "yes, the statement holds"
        ),
    ]


def laya_to_internal(qdef: dict[str, Any]) -> dict[str, Any]:
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    ins = qdef["instructions"]
    if not isinstance(ins, str):
        ins = json.dumps(ins)
    return {"t": t, "ins": ins, "crit": crit}


def laya_build_sequence(tok, state, q, max_len=512, head_max_len=192):  # type: ignore[no-untyped-def]
    mask_tok = tok.mask_token
    opts = laya_render_options(q)
    order = list(range(len(opts)))
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_ids = tok("%s question: %s" % (q["t"], ins), add_special_tokens=False)["input_ids"]
    opt_ids = []
    for i in order:
        opt_ids.append(
            [tok.mask_token_id]
            + tok(" " + opts[i].replace(mask_tok, " "), add_special_tokens=False)["input_ids"][:48]
        )
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[: max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    room = max(0, max_len - len(ids) - 1)
    state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    st = tok(state_text.replace(mask_tok, " "), add_special_tokens=False)["input_ids"]
    st = st[:room]
    ids = ids + st + [tok.sep_token_id]
    return ids[:max_len], [m for m in markers if m < max_len]


def laya_encode(tok, state, qdef, max_len, head_max_len):  # type: ignore[no-untyped-def]
    """laya.Agent.system_one's per-question step, including its refusal."""
    q = laya_to_internal(qdef)
    seq, markers = laya_build_sequence(tok, state, q, max_len, head_max_len)
    if len(markers) != len(laya_render_options(q)):
        raise ValueError("options exceed head_max_len")
    return seq, markers, QTYPES[q["t"]]


# ---- Julia ------------------------------------------------------------------------------


def julia_rows(state, questions):  # type: ignore[no-untyped-def]
    rows, meta = [], []
    for qid, q in questions.items():
        kind, criteria = q.get("type"), q.get("criteria")
        if kind == "choice":
            if isinstance(criteria, list):
                keys = labels = [str(x) for x in criteria]
            elif isinstance(criteria, dict):
                keys = list(criteria)
                labels = [v if v else k for k, v in criteria.items()]
            else:
                raise ValueError("a choice question needs options")
        elif kind == "score":
            labels, keys = [str(x) for x in criteria], [str(i) for i in range(len(criteria))]
        else:
            keys = ["false", "true"]
            given = criteria or {}
            labels = [given.get(k) or k for k in keys]
        if not 2 <= len(labels) <= 20:
            raise ValueError("a question needs 2 to 20 options")
        rows.append(
            {
                "state": state,
                "question": str(q.get("instructions", "")),
                "type": kind,
                "options": labels,
            }
        )
        meta.append((qid, kind, keys, labels))
    return rows, meta


def julia_sequence(tok, row, max_length=8192, head_length=512, option_tokens=48):  # type: ignore[no-untyped-def]
    state = row["state"]
    if any(tok.mask_token in text for text in [state, row["question"], *row["options"]]):
        raise ValueError("the request contains the model's reserved marker token")

    def clean(text: str) -> str:
        return text.replace(tok.mask_token, " ")

    enc = tok.encode
    head = enc(f"{row['type']} question: {clean(row['question'])}")
    option_ids = [enc(" " + clean(x)) for x in row["options"]]
    if any(len(x) > option_tokens for x in option_ids):
        raise ValueError("an option is longer than the model's token limit")
    options = [[tok.mask_token_id] + x[:option_tokens] for x in option_ids]
    budget = head_length - sum(map(len, options))
    if budget < 16:
        per_option = max(4, (head_length - 16) // len(options))
        options = [x[:per_option] for x in options]
        budget = head_length - sum(map(len, options))
    if len(head) > budget or any(len(x) != len(y) + 1 for x, y in zip(options, option_ids)):
        raise ValueError("the question and options are too long for the model")
    ids = [tok.cls_token_id, *head[: max(8, budget)], tok.sep_token_id]
    markers = []
    for option in options:
        markers.append(len(ids))
        ids.extend(option)
    ids.append(tok.sep_token_id)
    state_ids = enc(clean(state))
    room = max_length - len(ids) - 1
    if room < 1 or len(state_ids) > room:
        raise ValueError("the state is too long")
    return {
        "ids": ids + state_ids + [tok.sep_token_id],
        "markers": markers,
        "qtype": QTYPES[row["type"]],
    }


# ---- Decider ----------------------------------------------------------------------------

NARROW, LETTERS = 10, "ABCDEFGHIJ"
ISOLATED = "{q}\nProposed answer: {level}\nDoes the proposed answer fit?"


def decider_txt(v: Any) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def decider_render_question(spec: dict[str, Any], max_choice: int) -> dict[str, Any]:
    t = spec.get("type", "choice")
    crit = spec.get("criteria", spec.get("options"))
    raw = spec.get("instructions", spec.get("question", ""))
    if t in ("noul", "bool") and raw in (None, ""):
        ins = "Which answer fits the context?"
    else:
        ins = decider_txt(raw)
    if not ins:
        raise ValueError("question without instructions")
    if t == "choice":
        if isinstance(crit, (list, tuple)):
            crit = {str(c): None for c in crit}
        if not isinstance(crit, dict) or not 2 <= len(crit) <= max_choice:
            raise ValueError("choice criteria")
        names = list(crit)
        opts = [n if crit[n] in (None, "") else f"{n}: {decider_txt(crit[n])}" for n in names]
    elif t == "score":
        names = list(range(len(crit)))
        opts = [f"{i}: {decider_txt(c)}" for i, c in enumerate(crit)]
    else:
        names = [False, True]
        c = crit if crit is not None else {}
        f, tr = c.get("false"), c.get("true")
        opts = [
            "no" if f in (None, "") else f"no: {decider_txt(f)}",
            "yes" if tr in (None, "") else f"yes: {decider_txt(tr)}",
        ]
    return {
        "question": ins,
        "options": opts,
        "type": "noul" if t == "bool" else t,
        "names": names,
        "legend": [decider_txt(c) for c in crit] if t == "score" and crit is not None else None,
    }


def decider_rows(rq: dict[str, Any], isolated_levels: bool):  # type: ignore[no-untyped-def]
    if isolated_levels and rq["type"] == "score":
        return "iso", [
            (
                ISOLATED.format(q=rq["question"], level=re.sub(r"^\s*-?\d+\s*:\s*", "", level)),
                ["no", "yes"],
            )
            for level in rq["legend"]
        ]
    return "list", [(rq["question"], list(rq["options"]))]


class DeciderPrompter:
    def __init__(self, tok, max_options: int) -> None:  # type: ignore[no-untyped-def]
        self.tok = tok
        self.max_options = max_options
        upper = string.ascii_uppercase
        ids: list[int] = []
        for label in list(upper) + [a + b for a in upper for b in upper]:
            t = tok.encode(label, add_special_tokens=False)
            if len(t) == 1:
                ids.append(t[0])
            if len(ids) == max_options:
                break
        self.label_ids = ids
        self.open_ids = tok.encode("\n(", add_special_tokens=False)

    def enc(self, s: str) -> list[int]:
        return list(self.tok.encode(s, add_special_tokens=False))

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

    def rows(self, state: str, questions: dict[str, Any], isolated: bool) -> list[list[list[int]]]:
        context = self.enc("Context:\n" + state)
        out = []
        for spec in questions.values():
            rq = decider_render_question(spec, self.max_options)
            _, rows = decider_rows(rq, isolated)
            out.append([context + self.question_piece(t, o) for t, o in rows])
        return out
