"""noulxp calibrate and answering at another calibration.json (SPEC.md 7.1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from noulxp import schemas
from noulxp.answers import softmax
from noulxp.calibrate import (
    LabelError,
    Labelled,
    calibrate,
    fit,
    fitted_calibration,
    leading,
    read_labelled,
    table,
    target,
)
from noulxp.calibration import Calibration, Readout
from noulxp.errors import PackageError, RequestError
from noulxp.request import Question, parse_questions
from test_conformance import TOY_REQUESTS, ToyNative, toy_package  # noqa: F401 - a fixture

# The toy scores a choice's options and a noul's apart, a score's levels alike.
TEMPERATURES = {"choice": 0.3, "score": 2.0, "noul": 0.2}
TYPES = ["choice", "score", "noul"]


def one(spec: dict[str, Any]) -> Question:
    return parse_questions({"q": spec})[0]


def test_labels_become_distributions() -> None:
    choice = one({"type": "choice", "criteria": ["refund", "cancel", "track"]})
    assert target(choice, "cancel") == [0, 1, 0]
    assert target(choice, {"refund": 3, "track": 1}) == [0.75, 0, 0.25]
    answer = {"type": "choice", "choice": "refund", "probabilities": {"refund": 0.5, "cancel": 0.5}}
    assert target(choice, answer) == [0.5, 0.5, 0]
    score = one({"type": "score", "criteria": ["low", "mid", "high"]})
    assert target(score, 2) == [0, 0, 1]
    assert target(score, "1") == [0, 1, 0]
    noul = one({"type": "noul"})
    assert target(noul, True) == [0, 1]
    assert target(noul, False) == [1, 0]
    assert target(noul, "true") == [0, 1]
    assert target(noul, 0.25) == [0.75, 0.25]
    assert target(noul, {"type": "noul", "noul": 0.4}) == pytest.approx([0.6, 0.4])


@pytest.mark.parametrize(
    ("spec", "label"),
    [
        ({"type": "choice", "criteria": ["a", "b"]}, "c"),
        ({"type": "choice", "criteria": ["a", "b"]}, True),
        ({"type": "choice", "criteria": ["a", "b"]}, 0),
        ({"type": "choice", "criteria": ["a", "b"]}, {"c": 1}),
        ({"type": "choice", "criteria": ["a", "b"]}, {"a": -1, "b": 2}),
        ({"type": "choice", "criteria": ["a", "b"]}, {"a": 0}),
        ({"type": "score", "criteria": ["low", "high"]}, 2),
        ({"type": "noul"}, 1.5),
        ({"type": "noul"}, "maybe"),
    ],
)
def test_labels_that_name_no_option_are_refused(spec: dict[str, Any], label: Any) -> None:
    with pytest.raises(LabelError):
        target(one(spec), label)


def test_a_labelled_file_names_the_line_it_breaks_on(tmp_path: Path) -> None:
    good = {"state": "s", "questions": {"a": {"type": "noul"}}, "labels": {"a": True}}
    bad = {**good, "labels": {"b": True}}
    path = tmp_path / "labels.jsonl"
    path.write_text(json.dumps(good) + "\n\n" + json.dumps(bad) + "\n")
    with pytest.raises(LabelError, match=r"line 3.*no question 'b'"):
        read_labelled(path)
    path.write_text(json.dumps(good) + "\n")
    assert read_labelled(path)[0].labels == {"a": [0.0, 1.0]}


def readout(questions: list[Question], logits: list[list[float]]) -> Readout:
    def at(calibration: Calibration) -> list[tuple[Question, list[float]]]:
        return [
            (q, softmax(z, calibration.temperature(q.type, len(z))))
            for q, z in zip(questions, logits, strict=True)
        ]

    return at


def synthetic(
    labels_at: dict[str, float] | None, n: int = 40
) -> tuple[list[Readout], list[Labelled]]:
    """Readouts of random scores, labelled at known temperatures (None: evenly)."""
    rng = np.random.default_rng(3)
    questions = {
        "choice": {"type": "choice", "criteria": ["a", "b", "c", "d"]},
        "score": {"type": "score", "criteria": ["low", "mid", "high"]},
        "noul": {"type": "noul"},
    }
    parsed = parse_questions(questions)
    readouts, cases = [], []
    for i in range(n):
        logits = [rng.normal(0.0, 3.0, len(q.keys)).tolist() for q in parsed]
        labels = {
            q.id: softmax(z, labels_at[q.type]) if labels_at else [1 / len(z)] * len(z)
            for q, z in zip(parsed, logits, strict=True)
        }
        readouts.append(readout(parsed, logits))
        cases.append(Labelled("s", questions, labels, i + 1))
    return readouts, cases


def test_the_fit_finds_the_temperatures_the_labels_were_made_at() -> None:
    truth = {"choice": 0.37, "score": 2.9, "noul": 41.0}
    fitted = fit(*synthetic(truth), TYPES)
    for kind, value in truth.items():
        assert fitted[kind]["temperature"] == pytest.approx(value, rel=2e-3)
        assert fitted[kind]["bound"] is None


def test_hard_labels_fit_to_how_often_the_model_is_right() -> None:
    readouts, cases = synthetic({"choice": 3.0, "score": 3.0, "noul": 3.0})
    hard = leading(cases)
    assert all(sorted(g) == [0.0] * (len(g) - 1) + [1.0] for c in hard for g in c.labels.values())
    soft_fit, hard_fit = fit(readouts, cases, TYPES), fit(readouts, hard, TYPES)
    for kind in TYPES:
        # The scores' own leading option is the label's here, so hard labels ask for more
        # confidence than the soft ones they came from.
        assert hard_fit[kind]["temperature"] < soft_fit[kind]["temperature"]


def test_answers_that_carry_nothing_are_fitted_flat() -> None:
    fitted = fit(*synthetic(None), TYPES)
    assert all(fitted[kind]["bound"] == "highest" for kind in TYPES)


def test_a_fitted_calibration_keeps_what_was_not_fitted() -> None:
    package = {
        "standard": "noulxp/0.1",
        "temperature": {"choice": 1.5, "noul": 2.0},
        "by_option_count": [
            {"type": "choice", "min": 3, "max": 5, "temperature": 1.7},
            {"type": "noul", "min": 2, "max": 2, "temperature": 1.9},
        ],
    }
    data = fitted_calibration(package, {"choice": 4.0}, {"fitted_by": "a test"})
    assert schemas.errors(data, "calibration") == []
    cal = Calibration(data)
    assert cal.temperature("choice", 4) == 4.0
    assert cal.temperature("noul", 2) == 1.9


def write_calibration(path: Path, temperatures: dict[str, float]) -> Path:
    path.write_text(json.dumps({"standard": "noulxp/0.1", "temperature": temperatures}))
    return path


def labels_from(model: Any, path: Path) -> Path:
    """The requests of the toy set, labelled with a model's own unrounded answers."""
    lines = []
    for case in TOY_REQUESTS:
        state, questions = case["request"]["state"], case["request"]["questions"]
        dists, _ = model.distributions(state, questions)
        labels = {q.id: dict(zip(q.keys, p, strict=True)) for q, p in dists}
        lines.append(json.dumps({"state": state, "questions": questions, "labels": labels}))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_readouts_answer_as_the_runtime_does(toy_package: Path) -> None:  # noqa: F811
    from noulxp.runtime import load

    model = load(toy_package, device="cpu")
    items = [(c["request"]["state"], c["request"]["questions"]) for c in TOY_REQUESTS]
    items.append(("x", {"bad": {"type": "choice", "criteria": ["a"] * 30}}))
    got = model.readouts(items)
    assert isinstance(got[-1], RequestError)
    other = Calibration({"temperature": TEMPERATURES})
    for (state, questions), read in zip(items[:-1], got[:-1], strict=True):
        assert not isinstance(read, Exception)
        for calibration in (model.calibration, other):
            want, _ = model.distributions(state, questions, calibration)
            mine = read(calibration)
            assert [q.id for q, _ in mine] == [q.id for q, _ in want]
            for (_, p), (_, w) in zip(mine, want, strict=True):
                assert np.allclose(p, w, atol=1e-6)


def test_a_runtime_answers_at_the_calibration_it_is_given(
    toy_package: Path,  # noqa: F811
    tokens: Any,
    tmp_path: Path,
) -> None:
    from noulxp.conformance import generate, replay
    from noulxp.package import open_package, sha256_file
    from noulxp.runtime import load

    generate(toy_package, ToyNative(tokens), TOY_REQUESTS, log=lambda *_: None)
    path = write_calibration(tmp_path / "fitted.json", TEMPERATURES)
    plain = load(toy_package, device="cpu")
    fitted = load(toy_package, device="cpu", calibration=path)
    state, questions = TOY_REQUESTS[0]["request"]["state"], TOY_REQUESTS[0]["request"]["questions"]
    assert fitted.predict(state, questions) != plain.predict(state, questions)
    want, _ = plain.distributions(state, questions, Calibration({"temperature": TEMPERATURES}))
    got, _ = fitted.distributions(state, questions)
    for (_, p), (_, w) in zip(got, want, strict=True):
        assert np.allclose(p, w, atol=1e-9)
    # The conformance file was recorded at the package's calibration, and is replayed at it.
    assert replay(open_package(toy_package), fitted, device="cpu")["passed"]
    assert fitted.describe()["calibration"]["sha256"] == sha256_file(path)
    assert plain.describe()["calibration"] == "package"


def test_a_calibration_that_breaks_the_schema_is_refused(
    toy_package: Path,  # noqa: F811
    tmp_path: Path,
) -> None:
    from noulxp.runtime import load

    path = write_calibration(tmp_path / "bad.json", {"choice": -1.0})
    with pytest.raises(PackageError):
        load(toy_package, device="cpu", calibration=path)


def test_calibrate_recovers_the_calibration_the_labels_were_made_at(
    toy_package: Path,  # noqa: F811
    tmp_path: Path,
) -> None:
    from noulxp.runtime import load

    made = load(
        toy_package, device="cpu", calibration=write_calibration(tmp_path / "t.json", TEMPERATURES)
    )
    labels = labels_from(made, tmp_path / "labels.jsonl")
    report = calibrate(toy_package, labels, test=labels, device="cpu", log=lambda *_: None)
    fitted = report["calibration"]["temperature"]
    for kind in ("choice", "noul"):
        assert fitted[kind] == pytest.approx(TEMPERATURES[kind], rel=2e-3)
    assert report["fitted"]["score"]["bound"] == "no effect"
    assert fitted["score"] == 1.0  # the package's, kept
    assert schemas.errors(report["calibration"], "calibration") == []
    before, after = report["labelled"]["before"]["all"], report["labelled"]["after"]["all"]
    assert after["kl"] < 1e-4 < before["kl"]
    assert after["accuracy"] == before["accuracy"]
    assert report["test"]["after"] == report["labelled"]["after"]
    assert "choice" in table(report)


def test_the_command_writes_the_file_outside_the_package(
    toy_package: Path,  # noqa: F811
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from noulxp.cli import main
    from noulxp.runtime import load

    made = load(
        toy_package, device="cpu", calibration=write_calibration(tmp_path / "t.json", TEMPERATURES)
    )
    labels = str(labels_from(made, tmp_path / "labels.jsonl"))
    inside = str(toy_package / "calibration.json")
    assert main(["calibrate", str(toy_package), labels, "--out", inside]) == 2
    out = tmp_path / "fitted.json"
    assert main(["calibrate", str(toy_package), labels, "--out", str(out), "--device", "cpu"]) == 0
    assert schemas.errors(json.loads(out.read_text()), "calibration") == []
    request = tmp_path / "request.json"
    request.write_text(json.dumps(TOY_REQUESTS[0]["request"]))
    capsys.readouterr()
    args = ["run", str(toy_package), "--request", str(request), "--device", "cpu"]
    assert main([*args, "--calibration", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["answers"]


def test_a_server_names_the_calibration_it_answers_with(
    toy_package: Path,  # noqa: F811
    tmp_path: Path,
) -> None:
    from noulxp.serving import load_models

    path = write_calibration(tmp_path / "fitted.json", TEMPERATURES)
    models = load_models([toy_package], device="cpu", log=lambda *_: None, calibration=path)
    entry = models[0].describe()
    assert entry["calibration"]["temperature"] == TEMPERATURES
    assert schemas.errors({"object": "list", "data": [entry]}, "models") == []
    plain = load_models([toy_package], device="cpu", log=lambda *_: None)
    assert plain[0].describe()["calibration"] is None
    with pytest.raises(ValueError, match="one package"):
        load_models([toy_package, toy_package], device="cpu", calibration=path)
