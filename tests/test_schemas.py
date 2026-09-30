"""The JSON Schemas accept what the converters write and refuse what the spec forbids."""

from __future__ import annotations

import pytest

from noulxp.conformance import DEFAULT_REQUESTS, read_jsonl
from noulxp.export import decider as decider_export
from noulxp.export import julia as julia_export
from noulxp.export import laya as laya_export
from noulxp.schemas import NAMES, errors, schema

SPECIAL = {"cls": "[CLS]", "sep": "[SEP]", "marker": "[MASK]", "pad": "[PAD]"}
SHA = "0" * 64


def test_every_schema_is_a_valid_schema():
    from jsonschema import Draft202012Validator

    for name in NAMES:
        Draft202012Validator.check_schema(schema(name))


def test_converter_outputs_validate(tokens):
    assert (
        errors(laya_export.template({"max_len": 512, "head_max_len": 192}, SPECIAL), "template")
        == []
    )
    assert errors(julia_export.template(SPECIAL), "template") == []
    assert (
        errors(
            laya_export.calibration(
                {"temperature": [1.2, 1.1, 1.0], "temperature_by_options": {"choice:11+": 0.1}}
            ),
            "calibration",
        )
        == []
    )
    labels = decider_export.labels(tokens, 20)
    assert errors(decider_export.prompt({"isolated_levels": True}, labels), "prompt") == []
    assert errors(decider_export.prompt({"isolated_levels": False}, labels), "prompt") == []
    assert errors(decider_export.calibration({"temperature": 1.1}), "calibration") == []


def test_request_set_lines_are_requests():
    for case in read_jsonl(DEFAULT_REQUESTS):
        assert errors(case["request"], "request") == [], case["id"]


def _manifest(**overrides):  # type: ignore[no-untyped-def]
    base = {
        "standard": "noulxp/0.1",
        "name": "x/y",
        "profile": "encoder-markers",
        "weights": {
            "path": "model.onnx",
            "sha256": SHA,
            "format": "onnx",
            "data": [{"path": "model.safetensors", "sha256": SHA}],
        },
        "tokenizer": {"path": "tokenizer.json", "sha256": SHA},
        "template": {"path": "template.json", "sha256": SHA},
        "calibration": {"path": "calibration.json", "sha256": SHA},
        "limits": {"max_tokens": 512, "max_options": 20},
    }
    base.update(overrides)
    return base


def test_manifest_rules():
    assert errors(_manifest(), "manifest") == []
    assert errors(_manifest(standard="noulxp/0.2"), "manifest") == []
    assert errors(_manifest(standard="noulxp/0.3"), "manifest")
    assert errors(_manifest(profile="causal-letters"), "manifest")  # needs prompt and gguf
    assert errors(_manifest(tokenizer={"path": "../tokenizer.json", "sha256": SHA}), "manifest")
    assert errors(_manifest(tokenizer={"path": "/etc/passwd", "sha256": SHA}), "manifest")
    no_template = _manifest()
    del no_template["template"]
    assert errors(no_template, "manifest")


@pytest.mark.parametrize(
    ("case", "ok"),
    [
        (
            {
                "id": "a",
                "request": {"questions": {"q": {}}},
                "expected": {"q": {"type": "noul", "probabilities": {"false": 0.4, "true": 0.6}}},
            },
            True,
        ),
        ({"id": "a", "request": {"questions": {"q": {}}}, "error": {"message": "refused"}}, True),
        ({"id": "a", "request": {"questions": {"q": {}}}}, False),
        (
            {
                "id": "a",
                "request": {"questions": {"q": {}}},
                "error": {"message": "x"},
                "expected": {"q": {"type": "noul", "probabilities": {"false": 0.4, "true": 0.6}}},
            },
            False,
        ),
        (
            {
                "id": "a",
                "request": {"questions": {"q": {}}},
                "expected": {"q": {"type": "noul", "probabilities": {"true": 1.2, "false": 0}}},
            },
            False,
        ),
    ],
)
def test_conformance_case_schema(case, ok):
    assert (errors(case, "conformance-case") == []) is ok
