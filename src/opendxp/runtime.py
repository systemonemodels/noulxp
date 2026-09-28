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
    **options: Any,
) -> Any:
    """The runtime for this package.

    `device` is "auto", "cpu", or a backend name: for encoder-markers an
    onnxruntime provider ("coreml", "cuda", "openvino", "qnn", "directml"), for
    causal-letters "gpu" (every layer offloaded when llama.cpp was built with a
    GPU backend: Metal, CUDA, HIP, SYCL, Vulkan).
    """
    package = package_dir if isinstance(package_dir, Package) else open_package(package_dir)
    if package.profile == "encoder-markers":
        from opendxp.profiles.encoder_markers import EncoderMarkersRuntime

        return EncoderMarkersRuntime(package, provider=device, threads=threads, **options)
    from opendxp.profiles.causal_letters import CausalLettersRuntime

    return CausalLettersRuntime(package, device=device, threads=threads, **options)
