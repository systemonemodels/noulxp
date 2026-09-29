"""What sharing the prompt's prefix does to a causal-letters package's answers.

A request's rows share most of their tokens: the chat template, the state, the
question, and for rotations everything up to the options. An engine can keep the
last row in the cache and decode only what the next row adds to it, instead of
decoding every row from empty memory as SPEC.md 6.6 requires. This works for any
layout (typed ones, rotations, content-free priors). It runs the package's
conformance file both ways, through the reference runtime's own prompt building
and answer assembly, and compares each with the file.

    python scripts/prefix_sharing.py PACKAGE [--device cpu|gpu] [--report prefix-sharing.json]
"""

from __future__ import annotations

import argparse
import ctypes
import json
import time
from typing import Any

import numpy as np

from opendxp.conformance import compare_question, expected_from_answer, read_jsonl
from opendxp.package import open_package
from opendxp.profiles.causal_letters import CausalLettersRuntime


class SharedPrefix(CausalLettersRuntime):
    """The reference runtime, decoding only the tokens a row does not share with the last one."""

    def __init__(self, package: Any, *, device: str, threads: int) -> None:
        super().__init__(package, device=device, threads=threads)
        lc = self.lc
        lc.llama_free(self.ctx)
        decode = self.prompt.decode
        cp = lc.llama_context_default_params()
        cp.n_ctx = cp.n_batch = self.n_ctx
        cp.n_ubatch = min(int(decode["ubatch"]), self.n_ctx)
        cp.n_seq_max = 1
        cp.flash_attn_type = {
            "auto": lc.LLAMA_FLASH_ATTN_TYPE_AUTO,
            "enabled": lc.LLAMA_FLASH_ATTN_TYPE_ENABLED,
            "disabled": lc.LLAMA_FLASH_ATTN_TYPE_DISABLED,
        }[decode["flash_attention"]]
        kv = lc.GGML_TYPE_F16 if decode["kv_cache"] == "f16" else lc.GGML_TYPE_F32
        cp.type_k = cp.type_v = kv
        if hasattr(cp, "op_offload"):
            cp.op_offload = self.gpu
        if hasattr(cp, "swa_full"):
            # A sliding-window model's cache keeps every position, so a row's suffix can be dropped.
            cp.swa_full = True
        cp.n_threads = cp.n_threads_batch = self.threads
        self.ctx = lc.llama_init_from_model(self.model, cp)
        self.memory = lc.llama_get_memory(self.ctx)
        lc.llama_memory_clear(self.memory, True)
        self.cached: list[int] = []
        self.decoded = self.tokens_total = self.restarts = 0

    def _decode(self, tokens: list[int], start: int) -> None:
        b = self.batch
        for i, t in enumerate(tokens):
            b.token[i] = t
            b.pos[i] = start + i
            b.n_seq_id[i] = 1
            b.seq_id[i][0] = 0
            b.logits[i] = i == len(tokens) - 1
        b.n_tokens = len(tokens)
        if self.lc.llama_decode(self.ctx, b) != 0:
            raise RuntimeError("llama_decode failed")

    def slot_logits(self, ids: list[int], label_ids: list[int]) -> np.ndarray:
        lc = self.lc
        # What the cache holds of this row already; its last token is always decoded, for logits.
        n, limit = 0, min(len(self.cached), len(ids) - 1)
        while n < limit and self.cached[n] == ids[n]:
            n += 1
        # Some caches (sliding-window attention) cannot drop a suffix: those rows start over.
        if n and not lc.llama_memory_seq_rm(self.memory, 0, n, -1):
            lc.llama_memory_clear(self.memory, True)
            n, self.restarts = 0, self.restarts + 1
        try:
            self._decode(ids[n:], n)
        except RuntimeError:
            if not n:
                raise
            lc.llama_memory_clear(self.memory, True)
            n, self.restarts = 0, self.restarts + 1
            self._decode(ids, 0)
        self.cached = list(ids)
        self.decoded += len(ids) - n
        self.tokens_total += len(ids)
        pointer = ctypes.cast(
            lc.llama_get_logits_ith(self.ctx, len(ids) - n - 1), ctypes.POINTER(ctypes.c_float)
        )
        row = np.ctypeslib.as_array(pointer, shape=(self.n_vocab,))
        return np.asarray(row[np.asarray(label_ids)], dtype=np.float64)


def run(runtime: Any, cases: list[dict[str, Any]]) -> dict[str, Any]:
    dps: list[float] = []
    passed = agree = total = 0
    started = time.time()
    for case in cases:
        request = case["request"]
        answers = runtime.predict(request["state"], request["questions"])["answers"]
        ok = True
        for qid, expected in case["expected"].items():
            got = expected_from_answer(answers[qid])["probabilities"]
            result = compare_question(expected, list(got), list(got.values()))
            dps.append(result["dp"])
            agree += result["argmax_ok"]
            total += 1
            ok = ok and result["ok"]
        passed += ok
    return {
        "cases_passed": f"{passed}/{len(cases)}",
        "argmax": f"{agree}/{total}",
        "max_abs_dp": round(max(dps), 6),
        "mean_abs_dp": round(sum(dps) / len(dps), 6),
        "seconds": round(time.time() - started, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--report")
    args = parser.parse_args()

    package = open_package(args.package)
    cases = [c for c in read_jsonl(package.file("conformance")) if "expected" in c]
    report: dict[str, Any] = {"device": args.device, "threads": args.threads}
    for label, cls in (("each row in full", CausalLettersRuntime), ("shared prefix", SharedPrefix)):
        runtime = cls(package, device=args.device, threads=args.threads)
        try:
            report[label] = run(runtime, cases)
            if isinstance(runtime, SharedPrefix) and runtime.tokens_total:
                report[label]["tokens_decoded"] = f"{runtime.decoded}/{runtime.tokens_total}"
                report[label]["rows_started_over"] = runtime.restarts
        finally:
            runtime.close()
        print(label, json.dumps(report[label]), flush=True)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
