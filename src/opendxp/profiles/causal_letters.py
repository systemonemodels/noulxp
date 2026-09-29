"""Profile `causal-letters`: a decoder LM reads answer-letter logits at an answer slot.

For each question the prompt is built from prompt.json alone (SPEC.md, section 6):
the context piece and the question piece are tokenised separately and
concatenated, the options are lettered with the package's labels, and the
answer is the softmax, at the calibrated temperature, of the logits of the
offered labels' tokens at the last position. Score questions either list their
levels as options, or ask one yes/no row per level ("isolated levels") and
normalise the "yes" probabilities over the levels. A 0.2 prompt may instead
lay out each question type on its own, in a chat template and in rotation
(profiles/typed.py).

The weights are GGUF, run by llama.cpp through llama-cpp-python's low-level API:
one row per decode, fresh memory for each row, logits only at the slot.
"""

from __future__ import annotations

import ctypes
import functools
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from opendxp.answers import answer, softmax
from opendxp.calibration import Calibration
from opendxp.errors import BackendUnavailable, PackageError, RequestError
from opendxp.package import Package
from opendxp.profiles.encoder_markers import render_option
from opendxp.profiles.typed import TypedBuilder, TypedLayouts, trie_tokens
from opendxp.request import Question, parse_questions
from opendxp.spec import at_least
from opendxp.text import STATE_JSON, fill, render_state
from opendxp.tokens import Tokens


@functools.cache
def _silence(lc: Any) -> Any:
    """Turn llama.cpp's logging off, once per process.

    llama.cpp holds a single log callback for the whole process. A callback kept
    by one model would be freed with it while llama.cpp still calls it, and the
    next log line from any other model would crash the process. This one is
    cached, so it lives as long as the process does.
    """
    quiet = lc.llama_log_callback(lambda level, text, data: None)
    lc.llama_log_set(quiet, ctypes.c_void_p(0))
    return quiet


ENCODINGS = ("joined", "split-at-label")
# What a 0.1 prompt lays out for every type, and what a typed 0.2 prompt uses instead.
SHARED_LAYOUT = ("question", "labels", "layouts", "score")
TYPED_ONLY = ("chat", "rotations")
# Decode settings that change the numbers llama.cpp computes (SPEC.md 6.6). A package
# declares the ones its model's own engine uses; these are llama.cpp's defaults.
DECODE_DEFAULTS = {"flash_attention": "auto", "kv_cache": "f16", "ubatch": 2048}


@dataclass
class Row:
    ids: list[int]
    count: int


class Prompt:
    """prompt.json, checked."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        if data.get("profile") != "causal-letters":
            raise PackageError("prompt.json is not a causal-letters prompt")
        context = data.get("context") or {}
        if not isinstance(context.get("template"), str) or "{state}" not in context["template"]:
            raise PackageError("prompt.json: context.template must contain {state}")
        self.context = context["template"]
        self.context_max = context.get("max_tokens")
        state = data.get("state") or {}
        if state.get("json", "compact") not in STATE_JSON:
            raise PackageError(f"prompt.json: state.json must be one of {', '.join(STATE_JSON)}")
        if state.get("messages", "role-content") != "role-content":
            raise PackageError('prompt.json: state.messages can only be "role-content"')
        newer = [k for k in ("json", "messages", "empty") if k in state]
        newer += [k for k in ("types", *TYPED_ONLY) if k in data]
        if newer and not at_least(data.get("standard"), "odxp/0.2"):
            raise PackageError(f"prompt.json: {', '.join(newer)} need odxp/0.2")
        self.typed: TypedLayouts | None = None
        if "types" in data:
            mixed = [k for k in SHARED_LAYOUT if k in data]
            if mixed:
                raise PackageError(f"prompt.json: a typed prompt has no {', '.join(mixed)}")
            if self.context_max:
                raise PackageError(
                    "prompt.json: a typed prompt encodes each row whole, so it has no "
                    "context.max_tokens"
                )
            self.typed = TypedLayouts(data)
        else:
            if any(k in data for k in TYPED_ONLY):
                raise PackageError("prompt.json: chat and rotations need typed layouts (types)")
            self._one_layout(data)
        if data.get("slot", "last") != "last":
            raise PackageError('prompt.json: only the "last" answer slot is defined')
        self.decode = {**DECODE_DEFAULTS, **(data.get("decode") or {})}
        if self.decode["flash_attention"] not in ("auto", "enabled", "disabled"):
            raise PackageError(
                "prompt.json: decode.flash_attention must be auto, enabled or disabled"
            )
        if self.decode["kv_cache"] not in ("f16", "f32"):
            raise PackageError("prompt.json: decode.kv_cache must be f16 or f32")

    def _one_layout(self, data: dict[str, Any]) -> None:
        """The 0.1 layout: one question template and one list of labels for every type."""
        question = data.get("question") or {}
        for key in ("head", "option", "tail"):
            if not isinstance(question.get(key), str):
                raise PackageError(f"prompt.json: question.{key} must be a text template")
        self.head, self.option, self.tail = question["head"], question["option"], question["tail"]
        if self.option.count("{label}") != 1 or "{text}" not in self.option:
            raise PackageError("prompt.json: question.option needs one {label} and a {text}")
        self.before_label, self.after_label = self.option.split("{label}")
        self.labels = list(data.get("labels") or [])
        if len(self.labels) < 2 or len(set(self.labels)) != len(self.labels):
            raise PackageError("prompt.json: labels must be at least 2 distinct strings")
        self.layouts = sorted(data.get("layouts") or [], key=lambda x: int(x["max_options"]))
        if not self.layouts or any(x.get("encoding") not in ENCODINGS for x in self.layouts):
            raise PackageError("prompt.json: layouts must name an encoding for each option range")
        if int(self.layouts[-1]["max_options"]) > len(self.labels):
            raise PackageError("prompt.json: a layout allows more options than there are labels")
        self.score = dict(data.get("score") or {"mode": "list"})
        if self.score.get("mode") not in ("list", "isolated"):
            raise PackageError('prompt.json: score.mode must be "list" or "isolated"')
        if self.score["mode"] == "isolated":
            opts = self.score.get("options")
            read = self.score.get("read")
            if not isinstance(opts, list) or not isinstance(read, int) or not 0 <= read < len(opts):
                raise PackageError("prompt.json: isolated score needs options and a read index")
            self.strip = (
                re.compile(self.score["level_strip"]) if self.score.get("level_strip") else None
            )
        self.max_options = int(self.layouts[-1]["max_options"])

    def options_limit(self, kind: str) -> int:
        """The most options this prompt lays out for a question of this type."""
        if self.typed is not None:
            return self.typed.types[kind].max_options
        return self.max_options

    def instructions(self, question: Question) -> str:
        spec = self.data.get("instructions") or {}
        text = question.instructions or (spec.get("fallback") or {}).get(question.type, "")
        if not text and spec.get("required"):
            raise RequestError(f"question {question.id!r} needs instructions")
        return str(text)

    def layout(self, count: int) -> str:
        for layout in self.layouts:
            if count <= int(layout["max_options"]):
                return str(layout["encoding"])
        raise RequestError(f"this model reads at most {self.max_options} options")


class Builder:
    """Token ids for every row of a request."""

    def __init__(self, prompt: Prompt, tokens: Tokens) -> None:
        self.prompt, self.tokens = prompt, tokens
        self.label_ids = []
        for label in prompt.labels:
            single = tokens.single(label)
            if single is None:
                raise PackageError(f"label {label!r} is not a single token of this tokenizer")
            self.label_ids.append(single)
        if len(set(self.label_ids)) != len(self.label_ids):
            raise PackageError("two labels encode to the same token")

    def context(self, state: Any) -> list[int]:
        text = render_state(state, self.prompt.data.get("state"))
        ids = self.tokens.encode(fill(self.prompt.context, state=text))
        if self.prompt.context_max:
            ids = ids[: int(self.prompt.context_max)]
        return ids

    def question_piece(self, text: str, options: list[str]) -> list[int]:
        p, enc = self.prompt, self.tokens.encode
        head = fill(p.head, instructions=text)
        if p.layout(len(options)) == "joined":
            lines = "".join(
                fill(p.option, label=p.labels[j], text=o) for j, o in enumerate(options)
            )
            return enc(head + lines + p.tail)
        piece = enc(head)
        for j, o in enumerate(options):
            piece += [*enc(p.before_label), self.label_ids[j], *enc(fill(p.after_label, text=o))]
        return piece + enc(p.tail)

    def rows(self, question: Question) -> tuple[str, list[tuple[str, list[str]]]]:
        """("list" | "isolated", [(question text, option texts)]) for one question."""
        p = self.prompt
        text = p.instructions(question)
        if question.type == "score" and p.score["mode"] == "isolated":
            rows = []
            for option in question.options:
                level = option.description or ""
                if p.strip is not None:
                    level = p.strip.sub("", level, count=1)
                rows.append(
                    (
                        fill(p.score["question"], instructions=text, level=level),
                        list(p.score["options"]),
                    )
                )
            return "isolated", rows
        options = [render_option(p.data["options"], question, o) for o in question.options]
        return "list", [(text, options)]


def unique_tokens(context: int, pieces: list[list[int]]) -> int:
    """Input tokens counted as a prefix-sharing engine reads them: the context once,
    the common start of the question pieces once, then the rest of each piece."""
    if len(pieces) < 2:
        return context + sum(map(len, pieces))
    common, shortest, first = 0, min(map(len, pieces)), pieces[0]
    while common < shortest and all(x[common] == first[common] for x in pieces):
        common += 1
    return context + common + sum(len(x) - common for x in pieces)


Distributions = tuple[list[tuple[Question, list[float]]], dict[str, int]]
# The rows a request needs read, and how to finish it from their logits (in that order).
Plan = tuple[list[tuple[list[int], list[int]]], Callable[[list[np.ndarray]], Distributions]]


class CausalLettersRuntime:
    """The reference runtime: tokenizers + llama.cpp, nothing specific to any model."""

    profile = "causal-letters"
    # Rows decoded together per call (see __init__); one, the reference, unless asked.
    batch_rows = 1

    def __init__(
        self,
        package: Package,
        *,
        device: str = "auto",
        threads: int | None = None,
        max_tokens: int = 8192,
        batch_rows: int = 1,
    ) -> None:
        """`batch_rows` above 1 decodes up to that many rows together, each as its own
        sequence from empty memory, in one call: a serving choice, not part of the
        package, so a runtime using it is checked like any other (`opendxp check`)."""
        try:
            import llama_cpp as lc
        except ImportError as exc:
            raise BackendUnavailable("llama-cpp-python is not installed") from exc
        self.package = package
        self.prompt = Prompt(package.read_json("prompt"))
        self.tokens = Tokens(package.file("tokenizer"))
        self.builder: Builder | TypedBuilder = (
            TypedBuilder(self.prompt, self.tokens)
            if self.prompt.typed is not None
            else Builder(self.prompt, self.tokens)
        )
        self.calibration = Calibration(package.read_json("calibration"))
        self.rules = package.confidence_rules
        self.limits = package.limits
        self.lc = lc
        _silence(lc)
        lc.llama_backend_init()
        wanted = (device or "auto").lower()
        can_offload = bool(lc.llama_supports_gpu_offload())
        if wanted not in ("auto", "cpu") and not can_offload:
            raise BackendUnavailable(f"this llama.cpp build has no GPU backend for {device!r}")
        self.gpu = wanted != "cpu" and can_offload
        params = lc.llama_model_default_params()
        params.n_gpu_layers = -1 if self.gpu else 0
        started = time.perf_counter()
        self.model = lc.llama_model_load_from_file(str(package.file("weights")).encode(), params)
        if not self.model:
            raise PackageError("llama.cpp could not load the GGUF file")
        self.n_ctx = int(min(max_tokens, self.limits.get("max_tokens", max_tokens)))
        cp = lc.llama_context_default_params()
        # A whole row must fit one decode: llama.cpp aborts a decode larger than n_batch.
        decode = self.prompt.decode
        cp.n_ctx = self.n_ctx
        cp.n_batch = self.n_ctx
        cp.n_ubatch = min(int(decode["ubatch"]), self.n_ctx)
        self.batch_rows = max(1, int(batch_rows))
        cp.n_seq_max = self.batch_rows
        if self.batch_rows > 1:
            # The rows of one call share the context's cells, each row its own sequence.
            cp.kv_unified = True
        cp.flash_attn_type = {
            "auto": lc.LLAMA_FLASH_ATTN_TYPE_AUTO,
            "enabled": lc.LLAMA_FLASH_ATTN_TYPE_ENABLED,
            "disabled": lc.LLAMA_FLASH_ATTN_TYPE_DISABLED,
        }[decode["flash_attention"]]
        kv = lc.GGML_TYPE_F16 if decode["kv_cache"] == "f16" else lc.GGML_TYPE_F32
        cp.type_k = cp.type_v = kv
        self.threads = int(threads or min(8, os.cpu_count() or 4))
        cp.n_threads = cp.n_threads_batch = self.threads
        self.ctx = lc.llama_init_from_model(self.model, cp)
        if not self.ctx:
            raise PackageError("llama.cpp could not create a context")
        self.n_vocab = lc.llama_vocab_n_tokens(lc.llama_model_get_vocab(self.model))
        if max(self.builder.label_ids) >= self.n_vocab:
            raise PackageError("the tokenizer's label ids are outside the model's vocabulary")
        self.batch = lc.llama_batch_init(self.n_ctx, 0, 1)
        self.load_ms = (time.perf_counter() - started) * 1000

    def describe(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "backend": f"llama.cpp (llama-cpp-python {self.lc.__version__})",
            "gpu_offload": self.gpu,
            "threads": self.threads,
            "n_ctx": self.n_ctx,
            "decode": self.prompt.decode,
        }

    def slot_logits(self, ids: list[int], label_ids: list[int]) -> np.ndarray:
        """The logits of the tokens `label_ids` at the last position of `ids`."""
        lc = self.lc
        if len(ids) > self.n_ctx:
            raise RequestError(
                f"a prompt row is {len(ids)} tokens; this runtime reads {self.n_ctx}"
            )
        lc.llama_memory_clear(lc.llama_get_memory(self.ctx), True)
        b, last = self.batch, len(ids) - 1
        for i, t in enumerate(ids):
            b.token[i] = t
            b.pos[i] = i
            b.n_seq_id[i] = 1
            b.seq_id[i][0] = 0
            b.logits[i] = i == last
        b.n_tokens = len(ids)
        if lc.llama_decode(self.ctx, b) != 0:
            raise RuntimeError("llama_decode failed")
        pointer = ctypes.cast(
            lc.llama_get_logits_ith(self.ctx, last), ctypes.POINTER(ctypes.c_float)
        )
        row = np.ctypeslib.as_array(pointer, shape=(self.n_vocab,))
        return np.asarray(row[np.asarray(label_ids)], dtype=np.float64)

    def rows_logits(self, rows: list[tuple[list[int], list[int]]]) -> list[np.ndarray]:
        """`slot_logits` for many rows; with `batch_rows` above 1, several in each decode."""
        if self.batch_rows < 2:
            return [self.slot_logits(ids, read) for ids, read in rows]
        lc = self.lc
        out: list[np.ndarray | None] = [None] * len(rows)
        chunk: list[int] = []
        used = 0

        def flush() -> None:
            lc.llama_memory_clear(lc.llama_get_memory(self.ctx), True)
            b, k, lasts = self.batch, 0, []
            for seq, index in enumerate(chunk):
                ids = rows[index][0]
                for i, t in enumerate(ids):
                    b.token[k] = t
                    b.pos[k] = i
                    b.n_seq_id[k] = 1
                    b.seq_id[k][0] = seq
                    b.logits[k] = i == len(ids) - 1
                    k += 1
                lasts.append(k - 1)
            b.n_tokens = k
            if lc.llama_decode(self.ctx, b) != 0:
                raise RuntimeError("llama_decode failed")
            for index, last in zip(chunk, lasts, strict=True):
                pointer = ctypes.cast(
                    lc.llama_get_logits_ith(self.ctx, last), ctypes.POINTER(ctypes.c_float)
                )
                row = np.ctypeslib.as_array(pointer, shape=(self.n_vocab,))
                out[index] = np.asarray(row[np.asarray(rows[index][1])], dtype=np.float64)
            chunk.clear()

        for index, (ids, _read) in enumerate(rows):
            if len(ids) > self.n_ctx:
                raise RequestError(
                    f"a prompt row is {len(ids)} tokens; this runtime reads {self.n_ctx}"
                )
            if chunk and (len(chunk) >= self.batch_rows or used + len(ids) > self.n_ctx):
                flush()
                used = 0
            chunk.append(index)
            used += len(ids)
        if chunk:
            flush()
        return [x for x in out if x is not None]

    def checked(self, questions: Any) -> list[Question]:
        """The request's questions, each within the options this model reads."""
        parsed = parse_questions(questions)
        max_levels = self.limits.get("max_levels")
        for q in parsed:
            most = min(self.prompt.options_limit(q.type), self.limits.get("max_options", 1 << 30))
            if len(q.options) > most:
                raise RequestError(f"question {q.id!r} has more than {most} options")
            if max_levels and q.type == "score" and len(q.options) > max_levels:
                raise RequestError(f"question {q.id!r} has more than {max_levels} levels")
        return parsed

    def distributions(
        self, state: Any, questions: Any
    ) -> tuple[list[tuple[Question, list[float]]], dict[str, int]]:
        wanted, finish = self._plan(state, questions)
        return finish(self.rows_logits(wanted))

    def _plan(self, state: Any, questions: Any) -> Plan:
        """The rows a request needs read, and how to turn their logits into its answers."""
        wanted, finish = (
            self._typed_plan(state, questions)
            if isinstance(self.builder, TypedBuilder)
            else self._plain_plan(state, questions)
        )
        for ids, _label_ids in wanted:
            if len(ids) > self.n_ctx:
                raise RequestError(
                    f"a prompt row is {len(ids)} tokens; this runtime reads {self.n_ctx}"
                )
        return wanted, finish

    def _plain_plan(self, state: Any, questions: Any) -> Plan:
        parsed = self.checked(questions)
        planned = [(q, *self.builder.rows(q)) for q in parsed]
        context = self.builder.context(state)
        pieces: list[list[int]] = []
        for _q, _kind, rows in planned:
            for text, options in rows:
                pieces.append(self.builder.question_piece(text, options))
        wanted = [
            (context + pieces[n], self.builder.label_ids[: len(options)])
            for n, (_text, options) in enumerate(r for _q, _kind, rows in planned for r in rows)
        ]

        def finish(logits: list[np.ndarray]) -> Distributions:
            got = iter(logits)
            out = []
            for q, kind, rows in planned:
                t = self.calibration.temperature(q.type, len(q.options))
                probs = [softmax(next(got).tolist(), t) for _row in rows]
                if kind == "isolated":
                    level = int(self.prompt.score["read"])
                    fit = [row[level] for row in probs]
                    total = sum(fit) or 1e-9
                    out.append((q, [x / total for x in fit]))
                else:
                    out.append((q, probs[0]))
            return out, {"input_tokens": unique_tokens(len(context), pieces), "output_tokens": 0}

        return wanted, finish

    def _typed_plan(self, state: Any, questions: Any) -> Plan:
        """Typed layouts (SPEC.md 6.7, 6.8): a row per rotation, combined per option.

        A layout's rows are known before any is read, so they are collected first
        (the layout asked with placeholder logits) and combined once all are read.
        """
        builder = self.builder
        assert isinstance(builder, TypedBuilder)
        parsed = self.checked(questions)
        asked = [builder.ask(q) for q in parsed]
        state_text = builder.state(state)
        wanted: list[tuple[list[int], list[int]]] = []

        def record(ids: list[int], label_ids: list[int]) -> np.ndarray:
            wanted.append((list(ids), list(label_ids)))
            return np.zeros(len(label_ids))

        # The recording pass must not leave its placeholder priors in the builder's cache
        # of content-free priors: it records into a copy, and the real pass fills the cache.
        cached = builder.probe_priors
        builder.probe_priors = dict(cached)
        try:
            for a in asked:
                builder.distribution(state_text, a, record, 1.0)
        finally:
            builder.probe_priors = cached

        def finish(logits: list[np.ndarray]) -> Distributions:
            done = {
                (tuple(ids), tuple(label)): row
                for (ids, label), row in zip(wanted, logits, strict=True)
            }

            def lookup(ids: list[int], label_ids: list[int]) -> np.ndarray:
                # A row not read yet (a prior the cache has dropped since): read it now.
                found = done.get((tuple(ids), tuple(label_ids)))
                return found if found is not None else self.slot_logits(ids, label_ids)

            out, read = [], []
            for q, a in zip(parsed, asked, strict=True):
                t = self.calibration.temperature(q.type, len(q.options))
                p, rows = builder.distribution(state_text, a, lookup, t)
                out.append((q, p))
                read += rows
            return out, {"input_tokens": trie_tokens(read), "output_tokens": 0}

        return wanted, finish

    def predict(self, state: Any, questions: Any, decimals: int | None = 4) -> dict[str, Any]:
        dists, usage = self.distributions(state, questions)
        return {
            "answers": {q.id: answer(q, p, self.rules, decimals) for q, p in dists},
            "usage": usage,
        }

    def predict_many(
        self, items: list[tuple[Any, Any]], decimals: int | None = 4
    ) -> list[dict[str, Any] | Exception]:
        """Several requests' rows read together (with `batch_rows`), each finished alone.

        A request that cannot be asked gets its error; the others are unaffected.
        """
        results: list[dict[str, Any] | Exception] = [RequestError("not answered") for _ in items]
        plans: list[tuple[int, int, int, Any]] = []
        rows: list[tuple[list[int], list[int]]] = []
        for i, (state, questions) in enumerate(items):
            try:
                wanted, finish = self._plan(state, questions)
            except (RequestError, ValueError, KeyError, TypeError) as exc:
                results[i] = exc
                continue
            plans.append((i, len(rows), len(wanted), finish))
            rows.extend(wanted)
        logits = self.rows_logits(rows) if rows else []
        for i, start, count, finish in plans:
            dists, usage = finish(logits[start : start + count])
            results[i] = {
                "answers": {q.id: answer(q, p, self.rules, decimals) for q, p in dists},
                "usage": usage,
            }
        return results

    def close(self) -> None:
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
