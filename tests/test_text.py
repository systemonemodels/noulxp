from opendxp.text import annotate_indices, as_text, fill, placeholders, state_text


def test_fill_is_single_pass_and_keeps_unknown_placeholders():
    assert fill("{a} and {b}", a="{b}", b="x") == "{b} and x"
    assert (
        fill("{type} question: {instructions}", type="choice") == "choice question: {instructions}"
    )
    assert placeholders("\n({label}) {text}") == {"label", "text"}


def test_as_text_renders_structured_criteria_as_json():
    assert as_text("plain") == "plain"
    assert as_text({"desc": "café", "n": 1}) == '{"desc": "café", "n": 1}'
    assert as_text([1, 2]) == "[1, 2]"
    assert as_text(0) == "0"


def test_state_text():
    assert state_text("hello") == "hello"
    assert state_text({"a": [1, 2]}) == '{"a": [1, 2]}'
    annotated = annotate_indices({"x": list(range(8))}, 8)
    assert annotated["x"][0] == {"_index": 0, "value": 0}
    assert annotate_indices([1, 2], 8) == [1, 2]
    assert state_text({"x": list(range(8))}, 8).startswith('{"x": [{"_index": 0, "value": 0}')
