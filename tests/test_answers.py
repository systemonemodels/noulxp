import math

import pytest

from opendxp.answers import answer, argmax, confidence, softmax
from opendxp.request import parse_question


def test_softmax_with_temperature():
    p = softmax([1.0, 2.0, 3.0], 2.0)
    e = [math.exp(x / 2) for x in (1.0, 2.0, 3.0)]
    assert p == pytest.approx([x / sum(e) for x in e])
    assert sum(softmax([1000.0, 999.0])) == pytest.approx(1.0)


def test_confidence_rules():
    p = [0.7, 0.2, 0.1]
    assert confidence("max-probability", p) == 0.7
    assert confidence("typesafe", p) == pytest.approx((3 * 0.7 - 1) / 2)
    h = -sum(x * math.log(x) for x in p)
    assert confidence("entropy", p) == pytest.approx(1 - h / math.log(3))
    assert confidence("typesafe-ordinal", [1.0, 0.0, 0.0]) == 1.0
    assert confidence("none", p) is None
    assert argmax([0.2, 0.4, 0.4]) == 1


def test_answer_formats():
    choice = parse_question("c", {"type": "choice", "instructions": "", "criteria": ["a", "b"]})
    out = answer(choice, [0.25, 0.75], {"choice": "max-probability"})
    assert out == {
        "type": "choice",
        "choice": "b",
        "probabilities": {"a": 0.25, "b": 0.75},
        "confidence": 0.75,
    }
    score = parse_question(
        "s", {"type": "score", "instructions": "", "criteria": ["lo", "mid", "hi"]}
    )
    out = answer(score, [0.2, 0.3, 0.5])
    assert out["score"] == pytest.approx(1.3) and out["legend"] == {
        "0": "lo",
        "1": "mid",
        "2": "hi",
    }
    noul = parse_question("n", {"type": "noul", "instructions": "x"})
    assert answer(noul, [0.123456, 0.876544]) == {"type": "noul", "noul": 0.8765}
