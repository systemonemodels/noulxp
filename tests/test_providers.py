from __future__ import annotations

from pathlib import Path

import pytest

from opendxp.errors import BackendUnavailable
from opendxp.providers import ort_session


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
