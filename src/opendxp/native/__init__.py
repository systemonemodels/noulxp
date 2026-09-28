"""The models' own code, used only to write conformance files.

A conformance file records what a model's own runtime answers, so the runtime
here must be that code and nothing rewritten for OpenDXP:

    laya     the `laya` package (laya.load(...).system_one), Apache-2.0
    julia    opendxp.native.julia: the `julia` package's inference path, rebuilt
             without its transformers 5.0 pin (identical answers on the
             authors' 100 parity cases)
    decider  opendxp.native.decider: decider-ai's engine_gguf readout of the
             official GGUF through llama.cpp, every row decoded in full

Everything runs on the CPU: that is the reference.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

NATIVE_RUNTIMES = ("laya", "julia", "decider")


@dataclass
class NativeModel:
    name: str
    predict_fn: Any
    close_fn: Any = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def predict(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = self.predict_fn(state, questions)
        return result

    def close(self) -> None:
        if self.close_fn is not None:
            self.close_fn()


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256(path: str | None) -> str | None:
    if not path or not os.path.isfile(path):
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_native(
    runtime: str,
    checkpoint: str | Path,
    *,
    threads: int = 4,
) -> NativeModel:
    checkpoint = Path(checkpoint)
    if runtime == "laya":
        import laya

        agent = laya.load(str(checkpoint), device="cpu")
        return NativeModel(
            "laya",
            agent.system_one,
            provenance={
                "runtime": "laya",
                "code": "laya.load(...).system_one",
                "laya": _version("laya"),
                "torch": _version("torch"),
                "transformers": _version("transformers"),
                "device": "cpu",
            },
        )
    if runtime == "julia":
        from opendxp.native import julia

        model = julia.Julia(checkpoint, threads=threads)
        return NativeModel(
            "julia",
            model.system_one,
            provenance={
                "runtime": "julia",
                "code": "opendxp.native.julia.Julia",
                "code_sha256": _sha256(julia.__file__),
                **{
                    lib: _version(lib)
                    for lib in ("torch", "transformers", "tokenizers", "safetensors")
                },
                "threads": threads,
                "device": "cpu",
            },
        )
    if runtime == "decider":
        from opendxp.native import decider

        gguf_model = decider.Decider(checkpoint, threads=threads)
        return NativeModel(
            "decider",
            gguf_model.system_one,
            gguf_model.close,
            provenance={
                "runtime": "decider",
                "code": "opendxp.native.decider.Decider",
                "code_sha256": _sha256(decider.__file__),
                **{
                    lib: _version(lib)
                    for lib in ("llama_cpp_python", "transformers", "tokenizers", "numpy")
                },
                "decoding": "each row in full",
                "threads": threads,
                "device": "cpu",
            },
        )
    raise ValueError(f"unknown native runtime {runtime!r}; expected one of {NATIVE_RUNTIMES}")
