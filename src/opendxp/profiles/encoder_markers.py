"""Profile `encoder-markers`: a bidirectional encoder scores one marker token per option.

The input for one question is built from template.json alone (SPEC.md, section 5):
the pieces of the layout in order, each option preceded by the marker token, the
question block fitted to the head budget, and the state given whatever room the
total budget leaves. The graph (ONNX) takes

    input_ids [B, T] int64, attention_mask [B, T] int64,
    marker_positions [B, K] int64, marker_mask [B, K] bool, question_type [B] int64

and returns option_logits [B, K] float32. The answer is the softmax of the valid
logits divided by the calibrated temperature.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from opendxp.answers import answer, softmax
from opendxp.calibration import Calibration, Readout, described
from opendxp.errors import PackageError, RequestError
from opendxp.package import Package
from opendxp.providers import STATIC_SHAPE_PROVIDERS, choose_onnx_providers, ort_session
from opendxp.request import Option, Question, parse_questions
from opendxp.text import fill, state_text
from opendxp.tokens import Tokens

INPUTS = ("input_ids", "attention_mask", "marker_positions", "marker_mask", "question_type")
OUTPUT = "option_logits"
INPUT_TYPES = {
    "input_ids": "tensor(int64)",
    "attention_mask": "tensor(int64)",
    "marker_positions": "tensor(int64)",
    "marker_mask": "tensor(bool)",
    "question_type": "tensor(int64)",
}
PARTS = ("head", "options", "state")
BUDGET_DEFAULTS = {"reserve": 16, "option_floor": 4, "head_floor": 8}


@dataclass
class Encoded:
    ids: list[int]
    markers: list[int]
    qtype: int
    truncated: bool = False


@dataclass
class Template:
    """template.json, checked."""

    data: dict[str, Any]
    special: dict[str, str] = field(init=False)
    type_ids: dict[str, int] = field(init=False)
    layout: list[tuple[str, str]] = field(init=False)
    budgets: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        d = self.data
        if d.get("profile") != "encoder-markers":
            raise PackageError("template.json is not an encoder-markers template")
        self.special = dict(d.get("special_tokens") or {})
        for role in ("marker", "pad"):
            if not self.special.get(role):
                raise PackageError(f"template.json: special_tokens.{role} is required")
        self.type_ids = {k: int(v) for k, v in (d.get("question_type_ids") or {}).items()}
        if set(self.type_ids) != {"choice", "score", "noul"}:
            raise PackageError("template.json: question_type_ids must map choice, score and noul")
        self.layout = []
        for item in d.get("layout") or []:
            if "token" in item and item["token"] in self.special:
                self.layout.append(("token", item["token"]))
            elif item.get("part") in PARTS:
                self.layout.append(("part", item["part"]))
            else:
                raise PackageError(f"template.json: bad layout item {item!r}")
        parts = [name for kind, name in self.layout if kind == "part"]
        if sorted(parts) != sorted(PARTS):
            raise PackageError("template.json: the layout must hold head, options and state once")
        self.budgets = {**BUDGET_DEFAULTS, **{k: int(v) for k, v in d["budgets"].items()}}
        for key in ("total", "head", "option"):
            if self.budgets.get(key, 0) < 1:
                raise PackageError(f"template.json: budgets.{key} must be a positive integer")
        if d.get("truncation") not in ("strict", "truncate-state"):
            raise PackageError('template.json: truncation must be "strict" or "truncate-state"')
        if d.get("reserved_text", "replace") not in ("replace", "reject"):
            raise PackageError('template.json: reserved_text must be "replace" or "reject"')
        if not isinstance(d.get("head"), str):
            raise PackageError("template.json: head must be a text template")

    @property
    def strict(self) -> bool:
        return bool(self.data["truncation"] == "strict")

    @property
    def reject_reserved(self) -> bool:
        return bool(self.data.get("reserved_text", "replace") == "reject")

    def instructions(self, question: Question) -> str:
        spec = self.data.get("instructions") or {}
        text = question.instructions or (spec.get("fallback") or {}).get(question.type, "")
        if not text and spec.get("required"):
            raise RequestError(f"question {question.id!r} needs instructions")
        return str(text)

    def option_text(self, question: Question, option: Option) -> str:
        return render_option(self.data["options"], question, option)


def render_option(spec: dict[str, Any], question: Question, option: Option) -> str:
    """An option's text from the per-type forms: "described" when it has a description, else "bare".

    Noul declares a pair of forms for each of "false" and "true".
    """
    forms = spec.get(question.type)
    if question.type == "noul" and forms is not None:
        forms = forms.get(option.key)
    if not isinstance(forms, dict):
        raise PackageError(f"no option form for {question.type} questions")
    form = forms.get("described" if option.description is not None else "bare")
    if form is None:
        raise RequestError(
            f"question {question.id!r}: option {option.key!r} needs a description for this model"
        )
    return fill(
        form, name=option.name, description=option.description or "", index=str(option.index)
    )


def encode_question(template: Template, tokens: Tokens, question: Question, state: Any) -> Encoded:
    """The token ids and marker positions for one question (SPEC.md 5.3)."""
    t, b = template, template.budgets
    marker_text = t.special["marker"]
    head_text = fill(t.data["head"], type=question.type, instructions=t.instructions(question))
    option_texts = [t.option_text(question, o) for o in question.options]
    state_txt = state_text(state)
    if t.reject_reserved:
        if any(marker_text in x for x in (head_text, *option_texts, state_txt)):
            raise RequestError(f"the request contains the model's reserved token {marker_text!r}")
    else:
        head_text = head_text.replace(marker_text, " ")
        option_texts = [x.replace(marker_text, " ") for x in option_texts]
        state_txt = state_txt.replace(marker_text, " ")

    marker = tokens.id(marker_text)
    prefix = t.data["options"].get("prefix", "")
    head_ids = tokens.encode(head_text)
    raw = [tokens.encode(prefix + x) for x in option_texts]
    if t.strict and any(len(x) > b["option"] for x in raw):
        raise RequestError(f"an option is longer than this model's {b['option']}-token limit")
    options = [[marker, *x[: b["option"]]] for x in raw]
    budget = b["head"] - sum(map(len, options))
    if budget < b["reserve"]:
        per = max(b["option_floor"], (b["head"] - b["reserve"]) // max(1, len(options)))
        options = [x[:per] for x in options]
        budget = b["head"] - sum(map(len, options))
    cut = any(len(x) != len(y) + 1 for x, y in zip(options, raw, strict=True))
    if t.strict and (len(head_ids) > budget or cut):
        raise RequestError("the question and its options are too long for this model")
    head_ids = head_ids[: max(b["head_floor"], budget)]

    fixed = 0
    for kind, name in t.layout:
        if kind == "token":
            fixed += 1
        elif name == "head":
            fixed += len(head_ids)
        elif name == "options":
            fixed += sum(map(len, options))
    state_ids = tokens.encode(state_txt)
    room = b["total"] - fixed
    truncated = False
    if t.strict:
        if room < 1 or len(state_ids) > room:
            raise RequestError(
                f"the state is too long: this model reads up to {b['total']} tokens in all"
            )
    elif len(state_ids) > max(0, room):
        state_ids, truncated = state_ids[: max(0, room)], True

    ids: list[int] = []
    markers: list[int] = []
    for kind, name in t.layout:
        if kind == "token":
            ids.append(tokens.id(t.special[name]))
        elif name == "head":
            ids.extend(head_ids)
        elif name == "options":
            for option in options:
                markers.append(len(ids))
                ids.extend(option)
        else:
            ids.extend(state_ids)
    if len(ids) > b["total"]:
        ids, truncated = ids[: b["total"]], True
        if any(m >= b["total"] for m in markers):
            raise RequestError("the options do not fit in this model's input")
    return Encoded(ids, markers, t.type_ids[question.type], truncated)


def pack(encoded: list[Encoded], pad_id: int, multiple: int = 1) -> dict[str, np.ndarray]:
    """Right-padded tensors for the graph: padding has attention 0 and markers masked out."""
    n = len(encoded)
    length = max(len(e.ids) for e in encoded)
    length = -(-length // multiple) * multiple
    count = max(len(e.markers) for e in encoded)
    input_ids = np.full((n, length), pad_id, dtype=np.int64)
    attention = np.zeros((n, length), dtype=np.int64)
    positions = np.zeros((n, count), dtype=np.int64)
    mask = np.zeros((n, count), dtype=np.bool_)
    for i, e in enumerate(encoded):
        input_ids[i, : len(e.ids)] = e.ids
        attention[i, : len(e.ids)] = 1
        positions[i, : len(e.markers)] = e.markers
        mask[i, : len(e.markers)] = True
    qtype = np.asarray([e.qtype for e in encoded], dtype=np.int64)
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "marker_positions": positions,
        "marker_mask": mask,
        "question_type": qtype,
    }


class EncoderMarkersRuntime:
    """The reference runtime: tokenizers + onnxruntime, nothing specific to any model."""

    profile = "encoder-markers"
    # predict_many: the most rows, and padded tokens, one pass of the graph takes, and how
    # much longer than the shortest row in a pass the longest may be. Attention costs the
    # square of the padded length, so a short row padded to a long one's length costs more
    # than a pass of its own (Julia 1 on an A40: a pass of 1 to 16 short rows takes ~6 ms,
    # 32 rows 14 ms, 64 rows 42 ms).
    batch_rows = 32
    batch_tokens = 32 * 512
    batch_spread = 1.25
    batch_slack = 16

    def __init__(
        self,
        package: Package,
        *,
        provider: str = "auto",
        threads: int | None = None,
        optimization: str = "all",
        static_shapes: bool | None = None,
        precision: str = "fast",
    ) -> None:
        self.package = package
        self.precision = precision
        self.template = Template(package.read_json("template"))
        self.tokens = Tokens(package.file("tokenizer"))
        self.calibration = Calibration(package.read_json("calibration"))
        self.rules = package.confidence_rules
        self.limits = package.limits
        if self.template.budgets["total"] > self.limits.get("max_tokens", 1 << 30):
            raise PackageError("template budgets.total is over limits.max_tokens")
        self.pad_id = self.tokens.id(self.template.special["pad"])
        self.providers = choose_onnx_providers(provider)
        # Providers that compile for fixed shapes get one session per shape bucket.
        self.static = (
            bool(static_shapes)
            if static_shapes is not None
            else bool(STATIC_SHAPE_PROVIDERS & set(self.providers))
        )
        self._options = {
            "provider": provider,
            "threads": threads,
            "optimization": optimization,
            "precision": precision,
        }
        self._buckets: dict[tuple[int, int], Any] = {}
        started = time.perf_counter()
        if self.static:
            # Buckets compile on first use; a plain CPU session only reads the signature.
            self.session, _ = ort_session(
                package.file("weights"), provider="cpu", optimization="disable"
            )
            self._check_signature()
            self.session = None
        else:
            # The providers the session loaded, which is what describe() reports.
            self.session, self.providers = ort_session(
                package.file("weights"),
                provider=provider,
                threads=threads,
                optimization=optimization,
                precision=precision,
            )
            self._check_signature()
        self.load_ms = (time.perf_counter() - started) * 1000

    def _check_signature(self) -> None:
        found = {i.name: i.type for i in self.session.get_inputs()}
        if found != INPUT_TYPES:
            raise PackageError(f"the graph's inputs are {found}, not the standard signature")
        outputs = [o.name for o in self.session.get_outputs()]
        if OUTPUT not in outputs:
            raise PackageError(f"the graph has no {OUTPUT!r} output")

    def describe(self) -> dict[str, Any]:
        import onnxruntime as ort

        return {
            "profile": self.profile,
            "backend": f"onnxruntime {ort.__version__}",
            "providers": self.providers,
            "static_shapes": self.static,
            "precision": self.precision,
            "calibration": described(self),
        }

    def _bucket_session(self, tokens: int, options: int) -> Any:
        key = (tokens, options)
        if key not in self._buckets:
            # One compiled bucket at a time: a Core ML compilation holds a copy of the
            # weights in the temporary directory while its session lives.
            self._buckets.clear()
            gc.collect()
            session, _ = ort_session(
                self.package.file("weights"),
                fixed={"batch": 1, "tokens": tokens, "options": options},
                **self._options,
            )
            self._buckets[key] = session
        return self._buckets[key]

    def bucket(self, length: int, count: int) -> tuple[int, int]:
        """The fixed (tokens, options) shape a static-shape provider runs this input at."""
        total = self.template.budgets["total"]
        tokens = min(total, max(64, 1 << (length - 1).bit_length()))
        return tokens, max(count, int(self.limits.get("max_options", count)))

    def run(self, feeds: dict[str, np.ndarray]) -> np.ndarray:
        """option_logits for a packed batch; row by row at bucketed shapes when static."""
        if not self.static:
            return np.asarray(self.session.run([OUTPUT], feeds)[0])
        n, length = feeds["input_ids"].shape
        count = feeds["marker_positions"].shape[1]
        tokens, options = self.bucket(length, count)
        session = self._bucket_session(tokens, options)
        out = np.zeros((n, count), dtype=np.float32)
        for i in range(n):
            row = {
                "input_ids": np.full((1, tokens), self.pad_id, dtype=np.int64),
                "attention_mask": np.zeros((1, tokens), dtype=np.int64),
                "marker_positions": np.zeros((1, options), dtype=np.int64),
                "marker_mask": np.zeros((1, options), dtype=np.bool_),
                "question_type": feeds["question_type"][i : i + 1],
            }
            row["input_ids"][0, :length] = feeds["input_ids"][i]
            row["attention_mask"][0, :length] = feeds["attention_mask"][i]
            row["marker_positions"][0, :count] = feeds["marker_positions"][i]
            row["marker_mask"][0, :count] = feeds["marker_mask"][i]
            out[i] = session.run([OUTPUT], row)[0][0, :count]
        return out

    def encode(self, state: Any, questions: list[Question]) -> list[Encoded]:
        max_options = self.limits.get("max_options")
        max_levels = self.limits.get("max_levels")
        for q in questions:
            if max_options and len(q.options) > max_options:
                raise RequestError(f"question {q.id!r} has more than {max_options} options")
            if max_levels and q.type == "score" and len(q.options) > max_levels:
                raise RequestError(f"question {q.id!r} has more than {max_levels} levels")
        return [encode_question(self.template, self.tokens, q, state) for q in questions]

    def distributions(
        self, state: Any, questions: Any, calibration: Calibration | None = None
    ) -> tuple[list[tuple[Question, list[float]]], dict[str, int]]:
        """(question, calibrated probabilities) for every question, and the usage; at
        `calibration` if given, else at the runtime's."""
        calibration = calibration or self.calibration
        parsed = parse_questions(questions)
        encoded = self.encode(state, parsed)
        feeds = pack(encoded, self.pad_id)
        logits = self.run(feeds)
        out = []
        for i, (q, e) in enumerate(zip(parsed, encoded, strict=True)):
            k = len(e.markers)
            t = calibration.temperature(q.type, k)
            out.append((q, softmax(logits[i, :k].tolist(), t)))
        usage = {"input_tokens": sum(len(e.ids) for e in encoded), "output_tokens": 0}
        return out, usage

    def predict(self, state: Any, questions: Any, decimals: int | None = 4) -> dict[str, Any]:
        dists, usage = self.distributions(state, questions)
        answers = {q.id: answer(q, p, self.rules, decimals) for q, p in dists}
        return {"answers": answers, "usage": usage}

    def rows_logits(self, encoded: list[Encoded]) -> list[np.ndarray]:
        """option_logits for rows from any number of requests, run in as few passes as fit.

        Rows are sorted by length, and a pass holds at most `batch_rows` rows and
        `batch_tokens` padded tokens, of lengths within `batch_spread` times the shortest
        (plus `batch_slack` tokens): rows of similar length run together, and a long row
        never makes short ones pay for its length. Padding is masked, so a row's logits do
        not depend on the rows it shares a pass with.
        """
        out: list[np.ndarray] = [np.zeros(0, dtype=np.float32)] * len(encoded)
        chunk: list[int] = []

        def flush() -> None:
            logits = self.run(pack([encoded[i] for i in chunk], self.pad_id))
            for j, i in enumerate(chunk):
                out[i] = logits[j, : len(encoded[i].markers)]

        for i in sorted(range(len(encoded)), key=lambda i: len(encoded[i].ids)):
            longest = len(encoded[i].ids)  # sorted, so this row sets the pass's length
            if chunk and (
                len(chunk) >= self.batch_rows
                or (len(chunk) + 1) * longest > self.batch_tokens
                or longest > len(encoded[chunk[0]].ids) * self.batch_spread + self.batch_slack
            ):
                flush()
                chunk = []
            chunk.append(i)
        if chunk:
            flush()
        return out

    def predict_many(
        self, items: list[tuple[Any, Any]], decimals: int | None = 4
    ) -> list[dict[str, Any] | Exception]:
        """Several requests' questions read together: on a GPU, one pass instead of one each.

        A request that cannot be asked gets its error; the others are unaffected. A
        static-shape provider runs row by row anyway, so it answers them one at a time.
        """
        if self.static:
            results: list[dict[str, Any] | Exception] = []
            for state, questions in items:
                try:
                    results.append(self.predict(state, questions, decimals))
                except (RequestError, ValueError, KeyError, TypeError) as exc:
                    results.append(exc)
            return results
        results = [RequestError("not answered") for _ in items]
        plans: list[tuple[int, list[Question], int, int]] = []
        rows: list[Encoded] = []
        for i, (state, questions) in enumerate(items):
            try:
                parsed = parse_questions(questions)
                encoded = self.encode(state, parsed)
            except (RequestError, ValueError, KeyError, TypeError) as exc:
                results[i] = exc
                continue
            plans.append((i, parsed, len(rows), len(encoded)))
            rows.extend(encoded)
        logits = self.rows_logits(rows) if rows else []
        for i, parsed, start, count in plans:
            answers = {}
            for q, e, row in zip(
                parsed, rows[start : start + count], logits[start : start + count], strict=True
            ):
                t = self.calibration.temperature(q.type, len(e.markers))
                answers[q.id] = answer(q, softmax(row.tolist(), t), self.rules, decimals)
            tokens = sum(len(e.ids) for e in rows[start : start + count])
            results[i] = {"answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0}}
        return results

    def readouts(self, items: list[tuple[Any, Any]]) -> list[Readout | Exception]:
        """Each request read once, to be answered at any calibration (opendxp calibrate).

        A request that cannot be asked gets its error."""
        results: list[Readout | Exception] = []
        plans: list[tuple[list[Question], int, int] | Exception] = []
        rows: list[Encoded] = []
        for state, questions in items:
            try:
                parsed = parse_questions(questions)
                encoded = self.encode(state, parsed)
            except (RequestError, ValueError, KeyError, TypeError) as exc:
                plans.append(exc)
                continue
            plans.append((parsed, len(rows), len(encoded)))
            rows.extend(encoded)
        logits = [row.tolist() for row in self.rows_logits(rows)] if rows else []
        for plan in plans:
            if isinstance(plan, Exception):
                results.append(plan)
                continue
            parsed, start, count = plan
            results.append(_readout(parsed, logits[start : start + count]))
        return results

    def close(self) -> None:
        self.session = None
        self._buckets.clear()


def _readout(questions: list[Question], logits: list[list[float]]) -> Readout:
    def at(calibration: Calibration) -> list[tuple[Question, list[float]]]:
        return [
            (q, softmax(z, calibration.temperature(q.type, len(z))))
            for q, z in zip(questions, logits, strict=True)
        ]

    return at
