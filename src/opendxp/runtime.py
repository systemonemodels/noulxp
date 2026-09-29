"""Opening a package with the reference runtime its profile names."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from opendxp.package import Package, open_package


def load(
    package_dir: str | Path | Package,
    *,
    device: str = "auto",
    threads: int | None = None,
    calibration: str | Path | None = None,
    **options: Any,
) -> Any:
    """The runtime for this package.

    `device` is "auto", "cpu", or a backend name: for encoder-markers an
    onnxruntime provider ("coreml", "cuda", "openvino", "qnn", "directml"), for
    causal-letters "gpu" (every layer offloaded when llama.cpp was built with a
    GPU backend: Metal, CUDA, HIP, SYCL, Vulkan). `calibration` is a
    calibration.json to answer with in place of the package's (SPEC.md 7.1).
    """
    package = package_dir if isinstance(package_dir, Package) else open_package(package_dir)
    runtime: Any
    if package.profile == "encoder-markers":
        from opendxp.profiles.encoder_markers import EncoderMarkersRuntime

        runtime = EncoderMarkersRuntime(package, provider=device, threads=threads, **options)
    else:
        from opendxp.profiles.causal_letters import CausalLettersRuntime

        runtime = CausalLettersRuntime(package, device=device, threads=threads, **options)
    if calibration is not None:
        from opendxp.calibration import replace

        try:
            replace(runtime, calibration)
        except Exception:
            runtime.close()
            raise
    return runtime
