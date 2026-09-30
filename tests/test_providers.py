from __future__ import annotations

from pathlib import Path

import pytest

from noulxp.errors import BackendUnavailable
from noulxp.providers import ort_session


def test_a_named_provider_that_does_not_load_is_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ort = pytest.importorskip("onnxruntime")

    class Session:
        """onnxruntime when the CUDA provider's libraries are missing: the CPU only."""

        def __init__(self, path: str, options: object, providers: list[object]) -> None:
            self.asked = providers

        def get_providers(self) -> list[str]:
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(ort, "InferenceSession", Session)
    monkeypatch.setattr(
        ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    with pytest.raises(BackendUnavailable, match=r"CUDAExecutionProvider .* did not load"):
        ort_session(tmp_path / "model.onnx", provider="cuda")
    # "auto" takes the best that loads, and says which.
    _, loaded = ort_session(tmp_path / "model.onnx", provider="auto")
    assert loaded == ["CPUExecutionProvider"]


def test_encoder_export_refuses_an_old_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.metadata

    from noulxp.export.common import require_transformers

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "4.57.6")
    with pytest.raises(BackendUnavailable, match=r"transformers 4\.57\.6 .* 5\.2 or later"):
        require_transformers()
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "5.17.0")
    require_transformers()


def test_exact_precision_turns_tf32_off_on_cuda(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ort = pytest.importorskip("onnxruntime")
    asked: list[object] = []

    class Session:
        def __init__(self, path: str, options: object, providers: list[object]) -> None:
            asked.extend(providers)

        def get_providers(self) -> list[str]:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    monkeypatch.setattr(ort, "InferenceSession", Session)
    monkeypatch.setattr(
        ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    ort_session(tmp_path / "model.onnx", provider="cuda", precision="exact")
    assert ("CUDAExecutionProvider", {"use_tf32": "0"}) in asked
    asked.clear()
    ort_session(tmp_path / "model.onnx", provider="cuda")
    assert "CUDAExecutionProvider" in asked  # the provider's own defaults
    with pytest.raises(ValueError, match="precision"):
        ort_session(tmp_path / "model.onnx", provider="cuda", precision="half")
