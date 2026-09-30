"""Where a package runs: onnxruntime execution providers and llama.cpp GPU offload.

The engine chooses the device, not the model's author. "auto" takes CUDA when
this onnxruntime build has it, and the CPU otherwise; naming a provider
("coreml", "openvino", "qnn", "directml", ...) asks for it, with the CPU behind
it for any operator it cannot run. The others stay out of "auto" because they
compile a model per input shape: on an Apple M4, Julia 1 answered in 42 ms
through Core ML against 21 ms on the CPU, and compiling Laya's shape buckets
took minutes and gigabytes of temporary space (VALIDATION.md).
"""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any

from noulxp.errors import BackendUnavailable

ORT_PROVIDERS = {
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "qnn": "QNNExecutionProvider",
    "directml": "DmlExecutionProvider",
    "cpu": "CPUExecutionProvider",
}
# What "auto" tries, best first; the CPU is always the last resort. The other
# providers are used when named (see the module docstring).
PROVIDER_ORDER = ("cuda", "cpu")
# Options that make a provider usable for these graphs (dynamic shapes, float32 answers).
PROVIDER_OPTIONS: dict[str, dict[str, str]] = {
    "CoreMLExecutionProvider": {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"},
    "QNNExecutionProvider": {"backend_path": "QnnHtp.dll" if os.name == "nt" else "libQnnHtp.so"},
}
OPTIMIZATION = ("disable", "basic", "extended", "all")
# How exactly a GPU computes. "fast" is each backend's default; "exact" asks for float32 products
# where the default is a faster, less precise format. On an NVIDIA A40, ONNX Runtime's TF32
# moved Julia 1's probabilities by up to 0.0067 and float32 by 0.00005 (1-33 % slower), and
# llama.cpp's 16-bit cuBLAS accumulation moved AnyJev's by 0.049 and float32 by 0.008-0.014
# (half the speed). A serving choice, checked like any other (`noulxp check --precision`).
PRECISIONS = ("fast", "exact")
EXACT_OPTIONS: dict[str, dict[str, str]] = {"CUDAExecutionProvider": {"use_tf32": "0"}}
# Providers that compile a graph for fixed shapes. An engine pins the graph's symbolic
# dimensions ("batch", "tokens", "options"; SPEC.md 5.4) to a bucket and pads to it.
STATIC_SHAPE_PROVIDERS = frozenset({"CoreMLExecutionProvider", "QNNExecutionProvider"})


def available_onnx_providers() -> list[str]:
    try:
        import onnxruntime as ort
    except ImportError:
        return []
    return list(ort.get_available_providers())


def choose_onnx_providers(preference: str = "auto") -> list[str]:
    available = available_onnx_providers()
    if not available:
        raise BackendUnavailable("onnxruntime is not installed (pip install onnxruntime)")
    wanted = (preference or "auto").strip().lower()
    if wanted == "auto":
        return [ORT_PROVIDERS[k] for k in PROVIDER_ORDER if ORT_PROVIDERS[k] in available]
    name = ORT_PROVIDERS.get(wanted, preference)
    if name not in available:
        have = ", ".join(available)
        raise BackendUnavailable(f"{preference} is not in this onnxruntime build (have: {have})")
    return [name] if name == "CPUExecutionProvider" else [name, "CPUExecutionProvider"]


# ONNX's Attention operator (opset 23) is one fused node per attention. onnxruntime runs it
# on a CPU from 1.23 and on CUDA from 1.30; an older CUDA build would run those nodes on the
# CPU, inside a session that says CUDA.
FUSED_ATTENTION_OPSET = 23
FUSED_ATTENTION_CUDA = (1, 30)


def check_fused_attention(opset: int, providers: list[str], version: str) -> None:
    """Refuse a graph with fused attention on a provider that cannot run it."""
    if opset < FUSED_ATTENTION_OPSET or "CUDAExecutionProvider" not in providers:
        return
    have = tuple(int(x) for x in version.split(".")[:2] if x.isdigit())
    if have < FUSED_ATTENTION_CUDA:
        raise BackendUnavailable(
            f"this graph uses ONNX's Attention operator (opset {opset}), which onnxruntime runs "
            f"on CUDA from 1.30; this is {version}, which would run it on the CPU. Install "
            "onnxruntime-gpu>=1.30, or use a package exported at opset 18"
        )


def ort_session(
    path: Path,
    *,
    provider: str = "auto",
    threads: int | None = None,
    optimization: str = "all",
    fixed: dict[str, int] | None = None,
    precision: str = "fast",
) -> tuple[Any, list[str]]:
    import onnxruntime as ort

    if precision not in PRECISIONS:
        raise ValueError(f"precision is one of {', '.join(PRECISIONS)}, not {precision!r}")
    options = ort.SessionOptions()
    for dim, size in (fixed or {}).items():
        options.add_free_dimension_override_by_name(dim, int(size))
    if threads:
        options.intra_op_num_threads = int(threads)
        options.inter_op_num_threads = 1
    levels = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    options.graph_optimization_level = levels[optimization]
    options.log_severity_level = 3
    names = choose_onnx_providers(provider)
    providers: list[Any] = []
    for name in names:
        chosen = {
            **PROVIDER_OPTIONS.get(name, {}),
            **(EXACT_OPTIONS.get(name, {}) if precision == "exact" else {}),
        }
        providers.append((name, chosen) if chosen else name)
    session = ort.InferenceSession(str(path), options, providers=providers)
    # onnxruntime drops a provider it cannot load (a CUDA library of the wrong version, say)
    # with one log line and runs on the rest. A provider that was asked for by name and did
    # not load is an error, so that a check or a bench that says CUDA ran on CUDA.
    loaded = list(session.get_providers())
    missing = [n for n in names if n not in loaded]
    if missing and (provider or "auto").strip().lower() != "auto":
        raise BackendUnavailable(
            f"{missing[0]} is in this onnxruntime build but did not load, so it would run on "
            f"{', '.join(loaded)}: check the libraries it needs (for CUDA, the CUDA and cuDNN "
            "versions this onnxruntime release was built for)"
        )
    return session, loaded


def llama_gpu_offload() -> bool:
    try:
        import llama_cpp
    except ImportError:
        return False
    return bool(llama_cpp.llama_supports_gpu_offload())


def describe_machine() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": _cpu_name(),
        "cpus": os.cpu_count(),
        "python": platform.python_version(),
        "onnxruntime_providers": available_onnx_providers(),
    }
    import importlib.metadata
    import sys

    try:
        info["llama_cpp"] = importlib.metadata.version("llama_cpp_python")
    except importlib.metadata.PackageNotFoundError:
        info["llama_cpp"] = None
    if "llama_cpp" in sys.modules:  # importing it initialises every GPU backend: only if in use
        info["llama_gpu_offload"] = llama_gpu_offload()
    return info


def _cpu_name() -> str:
    if platform.system() == "Darwin":
        try:
            import subprocess

            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            )
            if out.stdout.strip():
                return out.stdout.strip()
        except OSError:
            pass
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"
