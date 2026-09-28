"""Profile `causal-letters`: a decoder LM reads answer-letter logits at an answer slot.

For each question the prompt is built from prompt.json alone (SPEC.md, section 6):
the context piece and the question piece are tokenised separately and
concatenated, the options are lettered with the package's labels, and the
answer is the softmax, at the calibrated temperature, of the logits of the
offered labels' tokens at the last position. Score questions either list their
levels as options, or ask one yes/no row per level ("isolated levels") and
normalise the "yes" probabilities over the levels.

The weights are GGUF, run by llama.cpp through llama-cpp-python's low-level API:
one row per decode, fresh memory for each row, logits only at the slot.
"""

from __future__ import annotations

import ctypes
import functools
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from opendxp.answers import answer, softmax
from opendxp.calibration import Calibration
from opendxp.errors import BackendUnavailable, PackageError, RequestError
from opendxp.package import Package
from opendxp.profiles.encoder_markers import render_option
from opendxp.request import Question, parse_questions
from opendxp.text import fill, state_text
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
        if data.get("slot", "last") != "last":
            raise PackageError('prompt.json: only the "last" answer slot is defined in odxp/0.1')
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
        self.decode = {**DECODE_DEFAULTS, **(data.get("decode") or {})}
        if self.decode["flash_attention"] not in ("auto", "enabled", "disabled"):
            raise PackageError(
                "prompt.json: decode.flash_attention must be auto, enabled or disabled"
            )
        if self.decode["kv_cache"] not in ("f16", "f32"):
            raise PackageError("prompt.json: decode.kv_cache must be f16 or f32")

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
        index_from = (self.prompt.data.get("state") or {}).get("index_arrays_from")
        ids = self.tokens.encode(fill(self.prompt.context, state=state_text(state, index_from)))
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


class CausalLettersRuntime:
    """The reference runtime: tokenizers + llama.cpp, nothing specific to any model."""

    profile = "causal-letters"

    def __init__(
        self,
        package: Package,
        *,
        device: str = "auto",
        threads: int | None = None,
        max_tokens: int = 8192,
    ) -> None:
        try:
            import llama_cpp as lc
        except ImportError as exc:
            raise BackendUnavailable("llama-cpp-python is not installed") from exc
        self.package = package
        self.prompt = Prompt(package.read_json("prompt"))
        self.tokens = Tokens(package.file("tokenizer"))
        self.builder = Builder(self.prompt, self.tokens)
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
        cp.n_seq_max = 1
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

    def slot_logits(self, ids: list[int], count: int) -> np.ndarray:
        """The logits of the first `count` labels at the last position of `ids`."""
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
        return np.asarray(row[np.asarray(self.builder.label_ids[:count])], dtype=np.float64)

    def distributions(
        self, state: Any, questions: Any
    ) -> tuple[list[tuple[Question, list[float]]], dict[str, int]]:
        parsed = parse_questions(questions)
        max_options = min(self.prompt.max_options, self.limits.get("max_options", 1 << 30))
        max_levels = self.limits.get("max_levels")
        planned = []
        for q in parsed:
            if len(q.options) > max_options:
                raise RequestError(f"question {q.id!r} has more than {max_options} options")
            if max_levels and q.type == "score" and len(q.options) > max_levels:
                raise RequestError(f"question {q.id!r} has more than {max_levels} levels")
            planned.append((q, *self.builder.rows(q)))
        context = self.builder.context(state)
        pieces: list[list[int]] = []
        for _q, _kind, rows in planned:
            for text, options in rows:
                pieces.append(self.builder.question_piece(text, options))
        out, n = [], 0
        for q, kind, rows in planned:
            t = self.calibration.temperature(q.type, len(q.options))
            probs = []
            for _text, options in rows:
                ids = context + pieces[n]
                n += 1
                probs.append(softmax(self.slot_logits(ids, len(options)).tolist(), t))
            if kind == "isolated":
                read = int(self.prompt.score["read"])
                fit = [row[read] for row in probs]
                total = sum(fit) or 1e-9
                out.append((q, [x / total for x in fit]))
            else:
                out.append((q, probs[0]))
        usage = {"input_tokens": unique_tokens(len(context), pieces), "output_tokens": 0}
        return out, usage

    def predict(self, state: Any, questions: Any, decimals: int | None = 4) -> dict[str, Any]:
        dists, usage = self.distributions(state, questions)
        return {
            "answers": {q.id: answer(q, p, self.rules, decimals) for q, p in dists},
            "usage": usage,
        }

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
