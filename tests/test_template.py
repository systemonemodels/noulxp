"""encoder-markers input construction against the models' own code (tests/oracles.py)."""

from __future__ import annotations

import random

import pytest

from generators import request
from noulxp.errors import PackageError, RequestError
from noulxp.export import julia as julia_export
from noulxp.export import laya as laya_export
from noulxp.profiles.encoder_markers import Template, encode_question, pack
from noulxp.request import parse_questions
from oracles import HFStyle, julia_rows, julia_sequence, laya_encode

LAYA_SPECIAL = {"cls": "[CLS]", "sep": "[SEP]", "marker": "[MASK]", "pad": "[PAD]"}
JULIA_SPECIAL = {"cls": "[CLS]", "sep": "[SEP]", "marker": "<mask>", "pad": "[PAD]"}


def ours(template: Template, tokens, state, questions):  # type: ignore[no-untyped-def]
    out = []
    for q in parse_questions(questions):
        try:
            e = encode_question(template, tokens, q, state)
            out.append((e.ids, e.markers, e.qtype))
        except RequestError:
            out.append("error")
    return out


@pytest.mark.parametrize(("total", "head"), [(512, 192), (1024, 256), (48, 24), (96, 40)])
def test_laya_template_reproduces_laya_build_sequence(tokens, total, head):
    template = Template(
        laya_export.template({"max_len": total, "head_max_len": head}, LAYA_SPECIAL)
    )
    tok = HFStyle(tokens, mask="[MASK]")
    rng = random.Random(total * 7 + head)
    compared = errors = 0
    for _ in range(250):
        req = request(rng, "[MASK]")
        mine = ours(template, tokens, req["state"], req["questions"])
        for (qid, spec), got in zip(req["questions"].items(), mine, strict=True):
            try:
                want = laya_encode(tok, req["state"], spec, total, head)
            except ValueError:
                want = "error"
            assert got == want, (qid, spec)
            compared += 1
            errors += want == "error"
    assert compared > 400
    if total < 64:
        assert errors > 0  # the smallest budget exercises Laya's refusal path


@pytest.mark.parametrize(("total", "head"), [(8192, 512), (2048, 512), (64, 32), (160, 48)])
def test_julia_template_reproduces_julia_sequence(tokens, total, head):
    template = Template(julia_export.template(JULIA_SPECIAL, total=total, head=head))
    tok = HFStyle(tokens, mask="<mask>")
    rng = random.Random(total + head)
    compared = errors = 0
    for _ in range(250):
        req = request(rng, "<mask>")
        mine = ours(template, tokens, req["state"], req["questions"])
        rows, _ = julia_rows(req["state"], req["questions"])
        for row, got in zip(rows, mine, strict=True):
            try:
                e = julia_sequence(tok, row, max_length=total, head_length=head)
                want = (e["ids"], e["markers"], e["qtype"])
            except ValueError:
                want = "error"
            assert got == want, row
            compared += 1
            errors += want == "error"
    assert compared > 400 and errors > 0


def test_layout_and_markers(tokens):
    template = Template(laya_export.template({"max_len": 64, "head_max_len": 32}, LAYA_SPECIAL))
    q = parse_questions(
        {"q": {"type": "choice", "instructions": "which", "criteria": ["yes", "no"]}}
    )
    e = encode_question(template, tokens, q[0], "the customer")
    cls, sep, mask = tokens.id("[CLS]"), tokens.id("[SEP]"), tokens.id("[MASK]")
    assert e.ids[0] == cls and e.ids[-1] == sep
    assert [e.ids[m] for m in e.markers] == [mask, mask]
    assert e.ids[e.markers[0] + 1] == tokens.id("yes")
    assert e.qtype == 0 and not e.truncated


def test_truncate_state_keeps_the_start(tokens):
    template = Template(laya_export.template({"max_len": 30, "head_max_len": 16}, LAYA_SPECIAL))
    q = parse_questions({"q": {"type": "noul", "instructions": "angry"}})[0]
    e = encode_question(template, tokens, q, " ".join(["the"] * 100))
    assert len(e.ids) == 30 and e.truncated


def test_reserved_text(tokens):
    strict = Template(julia_export.template(JULIA_SPECIAL, total=64, head=32))
    q = parse_questions({"q": {"type": "noul", "instructions": "is <mask> here"}})[0]
    with pytest.raises(RequestError):
        encode_question(strict, tokens, q, "state")
    lenient = Template(laya_export.template({"max_len": 64, "head_max_len": 32}, LAYA_SPECIAL))
    q = parse_questions({"q": {"type": "noul", "instructions": "is [MASK] here"}})[0]
    e = encode_question(lenient, tokens, q, "state [MASK]")
    assert e.ids.count(tokens.id("[MASK]")) == 2  # the two option markers only


def test_pack_pads_right(tokens):
    template = Template(laya_export.template({"max_len": 64, "head_max_len": 32}, LAYA_SPECIAL))
    qs = parse_questions(
        {
            "a": {"type": "noul", "instructions": "x"},
            "b": {"type": "choice", "instructions": "y", "criteria": ["a", "b", "c"]},
        }
    )
    encoded = [encode_question(template, tokens, q, "the") for q in qs]
    feeds = pack(encoded, tokens.id("[PAD]"))
    assert feeds["input_ids"].shape[0] == 2 and feeds["marker_mask"].shape == (2, 3)
    assert feeds["marker_mask"][0].tolist() == [True, True, False]
    assert feeds["attention_mask"][0].sum() == len(encoded[0].ids)
    assert feeds["question_type"].tolist() == [2, 0]


def test_template_validation():
    good = laya_export.template({"max_len": 64, "head_max_len": 32}, LAYA_SPECIAL)
    for broken in (
        {**good, "profile": "causal-letters"},
        {**good, "truncation": "sometimes"},
        {**good, "layout": good["layout"][:-2]},
        {**good, "special_tokens": {"cls": "[CLS]"}},
    ):
        with pytest.raises(PackageError):
            Template(broken)
