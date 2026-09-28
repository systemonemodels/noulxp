"""Where a package runs: onnxruntime execution providers and llama.cpp GPU offload.

The engine chooses the device, not the model's author. "auto" takes the first
available provider in PROVIDER_ORDER; naming one ("coreml", "cuda", ...) asks
for it, with the CPU behind it for any operator it cannot run.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any

from opendxp.errors import BackendUnavailable

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
# What "auto" tries, best first; the CPU is always the last resort.
PROVIDER_ORDER = ("cuda", "coreml", "openvino", "qnn", "directml", "cpu")
# Options that make a provider usable for these graphs (dynamic shapes, float32 answers).
PROVIDER_OPTIONS: dict[str, dict[str, str]] = {
    "CoreMLExecutionProvider": {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"},
    "QNNExecutionProvider": {"backend_path": "QnnHtp.dll" if os.name == "nt" else "libQnnHtp.so"},
}
OPTIMIZATION = ("disable", "basic", "extended", "all")
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


def ort_session(
    path: Path,
    *,
    provider: str = "auto",
    threads: int | None = None,
    optimization: str = "all",
    fixed: dict[str, int] | None = None,
) -> tuple[Any, list[str]]:
    import onnxruntime as ort

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
    providers = [(n, PROVIDER_OPTIONS[n]) if n in PROVIDER_OPTIONS else n for n in names]
    session = ort.InferenceSession(str(path), options, providers=providers)
    return session, names


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
