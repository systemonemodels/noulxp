"""How much llama.cpp backend settings move a causal-letters package's probabilities.

Decodes the rows of a few conformance cases on the CPU with the package's
declared settings (the reference), then again with other settings and on the
GPU, and reports the largest and mean difference in the letter probabilities.

    python scripts/llama_numerics.py PACKAGE [--cases r30 r33 r29 r21 r01]
"""

from __future__ import annotations

import argparse
import ctypes
import json

import numpy as np

from noulxp.answers import softmax
from noulxp.calibration import Calibration
from noulxp.conformance import read_jsonl
from noulxp.package import open_package
from noulxp.profiles.causal_letters import Builder, Prompt, _silence
from noulxp.request import parse_questions
from noulxp.tokens import Tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package")
    parser.add_argument("--cases", nargs="+", default=["r30", "r33", "r29", "r21", "r01"])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    import llama_cpp as lc

    package = open_package(args.package)
    builder = Builder(Prompt(package.read_json("prompt")), Tokens(package.file("tokenizer")))
    calibration = Calibration(package.read_json("calibration"))
    cases = {c["id"]: c for c in read_jsonl(package.file("conformance"))}
    rows = []
    for cid in args.cases:
        request = cases[cid]["request"]
        context = builder.context(request["state"])
        for q in parse_questions(request["questions"]):
            _, qrows = builder.rows(q)
            t = calibration.temperature(q.type, len(q.options))
            rows += [(context + builder.question_piece(x, o), len(o), t) for x, o in qrows]
    _silence(lc)
    lc.llama_backend_init()

    def run(gpu: bool, flash: int, kv: str) -> list[np.ndarray]:
        mp = lc.llama_model_default_params()
        mp.n_gpu_layers = -1 if gpu else 0
        model = lc.llama_model_load_from_file(str(package.file("weights")).encode(), mp)
        cp = lc.llama_context_default_params()
        cp.n_ctx = cp.n_batch = 8192
        cp.n_ubatch, cp.n_seq_max = 2048, 1
        cp.n_threads = cp.n_threads_batch = args.threads
        cp.flash_attn_type = flash
        if kv == "f32":
            cp.type_k = cp.type_v = lc.GGML_TYPE_F32
        ctx = lc.llama_init_from_model(model, cp)
        vocab = lc.llama_vocab_n_tokens(lc.llama_model_get_vocab(model))
        batch = lc.llama_batch_init(8192, 0, 1)
        out = []
        for ids, n, t in rows:
            lc.llama_memory_clear(lc.llama_get_memory(ctx), True)
            for i, token in enumerate(ids):
                batch.token[i], batch.pos[i], batch.n_seq_id[i] = token, i, 1
                batch.seq_id[i][0] = 0
                batch.logits[i] = i == len(ids) - 1
            batch.n_tokens = len(ids)
            if lc.llama_decode(ctx, batch) != 0:
                raise RuntimeError("llama_decode failed")
            pointer = ctypes.cast(
                lc.llama_get_logits_ith(ctx, len(ids) - 1), ctypes.POINTER(ctypes.c_float)
            )
            logits = np.ctypeslib.as_array(pointer, shape=(vocab,))
            out.append(np.asarray(softmax(logits[np.asarray(builder.label_ids[:n])].tolist(), t)))
        lc.llama_batch_free(batch)
        lc.llama_free(ctx)
        lc.llama_model_free(model)
        return out

    auto, off = lc.LLAMA_FLASH_ATTN_TYPE_AUTO, lc.LLAMA_FLASH_ATTN_TYPE_DISABLED
    reference = run(False, auto, "f16")
    variants = {"cpu, flash attention off": (False, off, "f16")}
    if lc.llama_supports_gpu_offload():
        variants |= {
            "gpu, declared settings": (True, auto, "f16"),
            "gpu, flash attention off": (True, off, "f16"),
            "gpu, flash attention off, f32 cache": (True, off, "f32"),
        }
    report = {
        "rows": len(rows),
        "reference": "cpu, declared settings (flash attention auto, f16 cache)",
    }
    for name, settings in variants.items():
        d = [float(np.abs(a - b).max()) for a, b in zip(run(*settings), reference, strict=True)]
        report[name] = {"max_abs_dp": max(d), "mean_max_abs_dp": float(np.mean(d))}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
