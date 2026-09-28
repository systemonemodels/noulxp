import pytest

from opendxp.errors import RequestError
from opendxp.request import parse_question, parse_request


def test_choice_dict_and_list():
    q = parse_question(
        "q", {"type": "choice", "instructions": "Pick", "criteria": {"a": "first", "b": None}}
    )
    assert [(o.key, o.description) for o in q.options] == [("a", "first"), ("b", None)]
    q = parse_question("q", {"type": "choice", "instructions": "Pick", "criteria": ["x", "y", "z"]})
    assert q.keys == ["x", "y", "z"] and all(o.description is None for o in q.options)


def test_score_levels_and_legend():
    q = parse_question("q", {"type": "score", "instructions": "How?", "criteria": ["low", "high"]})
    assert q.keys == ["0", "1"]
    assert q.legend() == {"0": "low", "1": "high"}


def test_noul_criteria_optional_and_partial():
    q = parse_question("q", {"type": "noul", "instructions": "Angry?"})
    assert q.keys == ["false", "true"] and q.options[1].description is None
    q = parse_question("q", {"type": "noul", "instructions": "", "criteria": {"true": "yes it is"}})
    assert q.options[0].description is None and q.options[1].description == "yes it is"


def test_structured_descriptions_become_json():
    q = parse_question(
        "q", {"type": "choice", "instructions": "", "criteria": {"a": {"k": 1}, "b": ""}}
    )
    assert q.options[0].description == '{"k": 1}' and q.options[1].description is None


@pytest.mark.parametrize(
    "spec",
    [
        {"type": "maybe", "criteria": ["a", "b"]},
        {"type": "choice", "criteria": ["a"]},
        {"type": "choice", "criteria": ["a", "a"]},
        {"type": "choice", "criteria": "a,b"},
        {"type": "score", "criteria": {"0": "low"}},
        {"type": "noul", "criteria": {"maybe": "x"}},
        {"type": "choice", "instructions": 3, "criteria": ["a", "b"]},
    ],
)
def test_invalid_questions_are_refused(spec):
    with pytest.raises(RequestError):
        parse_question("q", spec)


def test_request_shape():
    state, questions = parse_request({"state": None, "questions": {"q": {"type": "noul"}}})
    assert state == "" and questions[0].id == "q"
    with pytest.raises(RequestError):
        parse_request({"state": "x", "questions": {}})
