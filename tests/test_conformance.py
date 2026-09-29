"""Conformance comparison, coverage, and the whole check on a toy encoder-markers package.

The toy graph has the standard signature and scores each option by the id of
the token after its marker; a Python "native runtime" computes the same with
the Laya oracle's input construction, so the check exercises everything but
a real model.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import ClassVar

import pytest

from opendxp.conformance import (
    DEFAULT_REQUESTS,
    check,
    compare_question,
    coverage,
    expected_from_answer,
    generate,
    read_jsonl,
)
from opendxp.export import laya as laya_export
from opendxp.export.common import entry, manifest, place, write_json, write_manifest
from opendxp.spec import TOLERANCE
from oracles import HFStyle, laya_encode


def test_compare_within_and_over_tolerance():
    want = {"type": "choice", "probabilities": {"a": 0.6, "b": 0.4}}
    assert compare_question(want, ["a", "b"], [0.605, 0.395])["ok"]
    over = compare_question(want, ["a", "b"], [0.6 - 2 * TOLERANCE, 0.4 + 2 * TOLERANCE])
    assert not over["ok"] and over["argmax_ok"]
    flipped = compare_question(
        {"type": "choice", "probabilities": {"a": 0.52, "b": 0.48}}, ["a", "b"], [0.49, 0.51]
    )
    assert not flipped["argmax_ok"]


def test_ties_accept_either_argmax():
    want = {"type": "choice", "probabilities": {"a": 0.5, "b": 0.5}}
    assert compare_question(want, ["a", "b"], [0.499, 0.501])["ok"]
    near = {"type": "choice", "probabilities": {"a": 0.5004, "b": 0.4996}}
    assert compare_question(near, ["a", "b"], [0.4999, 0.5001])["argmax_ok"]


def test_keys_must_match():
    want = {"type": "choice", "probabilities": {"a": 0.5, "b": 0.5}}
    assert not compare_question(want, ["b", "a"], [0.5, 0.5])["ok"]


def test_expected_from_answers():
    assert expected_from_answer({"type": "noul", "noul": 0.8626}) == {
        "type": "noul",
        "probabilities": {"false": 0.1374, "true": 0.8626},
    }
    e = expected_from_answer(
        {"type": "score", "score": 1.2, "probabilities": {"0": 0.2, "1": 0.4, "2": 0.4}}
    )
    assert e["probabilities"] == {"0": 0.2, "1": 0.4, "2": 0.4}


def test_the_shipped_request_set_meets_the_minimums():
    cases = read_jsonl(DEFAULT_REQUESTS)
    cov = coverage(cases)
    assert cov["ok"], cov["checks"]
    assert {"ne", "th"} <= set(cov["languages"])
    assert max(cov["choice_option_counts"]) >= 12


# ---- the whole check on a toy package ---------------------------------------------------

LAYA_CFG = {
    "max_len": 96,
    "head_max_len": 40,
    "temperature": [2.0, 1.0, 1.5],
    "temperature_by_options": {},
}
SPECIAL = {"cls": "[CLS]", "sep": "[SEP]", "marker": "[MASK]", "pad": "[PAD]"}


def toy_graph(path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "tokens"]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "tokens"]),
        helper.make_tensor_value_info("marker_positions", TensorProto.INT64, ["batch", "options"]),
        helper.make_tensor_value_info("marker_mask", TensorProto.BOOL, ["batch", "options"]),
        helper.make_tensor_value_info("question_type", TensorProto.INT64, ["batch"]),
    ]
    output = helper.make_tensor_value_info("option_logits", TensorProto.FLOAT, ["batch", "options"])
    c = helper.make_tensor
    nodes = [
        helper.make_node("Constant", [], ["one"], value=c("one", TensorProto.INT64, [], [1])),
        helper.make_node("Add", ["marker_positions", "one"], ["next"]),
        helper.make_node("GatherElements", ["input_ids", "next"], ["token"], axis=1),
        helper.make_node("Cast", ["token"], ["tokenf"], to=TensorProto.FLOAT),
        helper.make_node("Constant", [], ["scale"], value=c("scale", TensorProto.FLOAT, [], [0.1])),
        helper.make_node("Mul", ["tokenf", "scale"], ["scaled"]),
        helper.make_node("Cast", ["attention_mask"], ["maskf"], to=TensorProto.FLOAT),
        helper.make_node("Constant", [], ["axes"], value=c("axes", TensorProto.INT64, [1], [1])),
        helper.make_node("ReduceSum", ["maskf", "axes"], ["length"], keepdims=1),
        helper.make_node("Constant", [], ["zero"], value=c("zero", TensorProto.FLOAT, [], [0.0])),
        helper.make_node("Mul", ["length", "zero"], ["nothing"]),
        helper.make_node("Cast", ["question_type"], ["qf"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["qf", "axes"], ["qcol"]),
        helper.make_node("Add", ["scaled", "qcol"], ["with_type"]),
        helper.make_node("Add", ["with_type", "nothing"], ["logits"]),
        helper.make_node(
            "Constant", [], ["floor"], value=c("floor", TensorProto.FLOAT, [], [-1e4])
        ),
        helper.make_node("Where", ["marker_mask", "logits", "floor"], ["option_logits"]),
    ]
    graph = helper.make_graph(nodes, "toy", inputs, [output])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


class ToyNative:
    """Laya's input construction (the oracle) and the toy graph's arithmetic, in Python."""

    provenance: ClassVar[dict[str, str]] = {"runtime": "toy"}

    def __init__(self, tokens) -> None:  # type: ignore[no-untyped-def]
        self.tok = HFStyle(tokens)
        self.cal = laya_export.calibration(LAYA_CFG)["temperature"]

    def predict(self, state, questions):  # type: ignore[no-untyped-def]
        answers = {}
        for qid, spec in questions.items():
            ids, markers, qtype = laya_encode(self.tok, state, spec, 96, 40)
            z = [0.1 * ids[m + 1] + qtype for m in markers]
            t = self.cal[spec["type"]]
            e = [math.exp((x - max(z)) / t) for x in z]
            p = [x / sum(e) for x in e]
            if spec["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": round(p[1], 4)}
            else:
                crit = spec["criteria"]
                keys = (
                    list(crit) if spec["type"] == "choice" else [str(i) for i in range(len(crit))]
                )
                answers[qid] = {
                    "type": spec["type"],
                    "probabilities": {k: round(x, 4) for k, x in zip(keys, p, strict=True)},
                }
        return {"answers": answers}


@pytest.fixture()
def toy_package(tmp_path: Path, tokenizer_file: Path) -> Path:
    pytest.importorskip("onnxruntime")
    out = tmp_path / "toy"
    out.mkdir()
    toy_graph(out / "model.onnx")
    place(tokenizer_file, out / "tokenizer.json", "copy")
    write_json(out / "template.json", laya_export.template(LAYA_CFG, SPECIAL))
    write_json(out / "calibration.json", laya_export.calibration(LAYA_CFG))
    files = {
        "weights": entry(out, "model.onnx", format="onnx", opset=18),
        "tokenizer": entry(out, "tokenizer.json"),
        "template": entry(out, "template.json"),
        "calibration": entry(out, "calibration.json"),
    }
    write_manifest(
        out,
        manifest(
            name="test/toy",
            profile="encoder-markers",
            files=files,
            limits={"max_tokens": 96, "max_options": 20},
            confidence={"choice": "entropy"},
            source={"model": "toy"},
        ),
    )
    return out


TOY_REQUESTS = [
    {
        "id": "a",
        "request": {
            "state": "the customer wants a refund",
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": "what",
                    "criteria": {"refund": None, "cancel": "order", "track": None},
                },
                "angry": {"type": "noul", "instructions": "angry"},
                "level": {
                    "type": "score",
                    "instructions": "how",
                    "criteria": ["low", "medium", "high"],
                },
            },
        },
    },
    {
        "id": "b",
        "request": {
            "state": "hello [MASK] " * 40,
            "questions": {
                "x": {"type": "choice", "instructions": "which", "criteria": ["yes", "no"]},
            },
        },
    },
]


def test_generate_then_check_passes(toy_package: Path, tokens):  # type: ignore[no-untyped-def]
    summary = generate(toy_package, ToyNative(tokens), TOY_REQUESTS, log=lambda *_: None)
    assert summary["cases"] == 2 and summary["questions"] == 4
    report = check(toy_package, device="cpu")
    assert report["passed"], report["failures"]
    assert report["max_abs_dp"] <= 5e-5 + 1e-9
    assert report["argmax_agreement"] == {"agree": 4, "total": 4}
    assert not report["compatible"]  # two cases are below the coverage minimum

    from opendxp.schemas import errors

    assert errors(report, "check-report") == []

    from opendxp.runtime import load

    model = load(toy_package, device="cpu")
    out = model.predict(
        "the customer", {"q": {"type": "choice", "instructions": "w", "criteria": ["yes", "no"]}}
    )
    assert errors(out, "response") == []
    assert set(out["answers"]["q"]) == {"type", "choice", "probabilities", "confidence"}


def test_replay_checks_a_runtime_that_is_already_loaded(toy_package: Path, tokens):  # type: ignore[no-untyped-def]
    from opendxp.conformance import replay
    from opendxp.package import open_package
    from opendxp.runtime import load

    generate(toy_package, ToyNative(tokens), TOY_REQUESTS, log=lambda *_: None)
    model = load(toy_package, device="cpu")
    report = replay(open_package(toy_package), model, device="cpu")
    fresh = check(toy_package, device="cpu")
    same = ("passed", "cases", "cases_passed", "max_abs_dp", "argmax_agreement", "errors")
    assert report["passed"] and report["cases"] == 2
    assert {k: report[k] for k in same} == {k: fresh[k] for k in same}
    # The runtime is still open, as an engine's must be.
    state = TOY_REQUESTS[0]["request"]["state"]
    assert model.predict(state, TOY_REQUESTS[0]["request"]["questions"])["answers"]


def test_predict_many_answers_as_predict_does(toy_package: Path):  # type: ignore[no-untyped-def]
    from opendxp.errors import RequestError
    from opendxp.runtime import load

    model = load(toy_package, device="cpu")
    model.batch_rows = 2  # several passes, rows from different requests sharing them
    items = [(r["request"]["state"], r["request"]["questions"]) for r in TOY_REQUESTS]
    items.insert(1, ("x", {"bad": {"type": "choice", "instructions": "w", "criteria": ["a"] * 30}}))
    got = model.predict_many(items)
    assert isinstance(got[1], RequestError)
    for (state, questions), mine in zip(items[:1] + items[2:], got[:1] + got[2:], strict=True):
        assert mine == model.predict(state, questions)


def test_check_fails_on_a_perturbed_expectation(toy_package: Path, tokens):  # type: ignore[no-untyped-def]
    generate(toy_package, ToyNative(tokens), TOY_REQUESTS, log=lambda *_: None)
    path = toy_package / "conformance.jsonl"
    cases = [json.loads(x) for x in path.read_text().splitlines()]
    probs = cases[0]["expected"]["intent"]["probabilities"]
    first = next(iter(probs))
    probs[first] = round(probs[first] + 0.05, 4)
    path.write_text("".join(json.dumps(c) + "\n" for c in cases))
    report = check(toy_package, device="cpu", hashes=False)
    assert not report["passed"]
    assert report["failures"][0]["id"] == "a"
    assert report["max_abs_dp"] == pytest.approx(0.05, abs=1e-4)


def test_expected_refusals_must_be_refused(toy_package: Path, tokens):  # type: ignore[no-untyped-def]
    requests = [
        *TOY_REQUESTS,
        {
            "id": "c",
            "request": {
                "state": "x",
                "questions": {"q": {"type": "choice", "criteria": ["a", "b"] * 11}},
            },
        },
    ]
    requests[2]["request"]["questions"]["q"]["criteria"] = [
        f"o{i}" for i in range(21)
    ]  # over max_options
    native = ToyNative(tokens)

    class Refusing(ToyNative):
        def predict(self, state, questions):  # type: ignore[no-untyped-def]
            if any(len(q.get("criteria") or []) > 20 for q in questions.values()):
                raise ValueError("too many options")
            return native.predict(state, questions)

    generate(toy_package, Refusing(tokens), requests, log=lambda *_: None)
    report = check(toy_package, device="cpu")
    assert report["passed"] and report["errors"] == {"expected": 1, "matched": 1}
