"""Graphs with ONNX's fused Attention operator (opset 23): the mask fix and the runtime guard."""

from __future__ import annotations

import numpy as np
import pytest

from noulxp.errors import BackendUnavailable
from noulxp.providers import check_fused_attention
from test_conformance import toy_package  # noqa: F401 - a fixture

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")


def attention_model(mask_shape: list[int]) -> object:
    """One Attention node: Q (1, 2, 3, 4) attends over K, V (1, 2, 5, 4), with a boolean mask."""
    from onnx import TensorProto, helper

    q = helper.make_tensor_value_info("q", TensorProto.FLOAT, [1, 2, 3, 4])
    k = helper.make_tensor_value_info("k", TensorProto.FLOAT, [1, 2, 5, 4])
    v = helper.make_tensor_value_info("v", TensorProto.FLOAT, [1, 2, 5, 4])
    m = helper.make_tensor_value_info("m", TensorProto.BOOL, mask_shape)
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, None)
    node = helper.make_node("Attention", ["q", "k", "v", "m"], ["y"])
    graph = helper.make_graph([node], "attention", [q, k, v, m], [y])
    # IR version 11 is the one opset 23 came with; newer onnx releases write newer ones.
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)], ir_version=11)


def reference(q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray) -> np.ndarray:
    scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(q.shape[-1])
    scores = np.where(np.broadcast_to(mask, scores.shape), scores, -np.inf)
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    return (weights / weights.sum(-1, keepdims=True)) @ v


@pytest.mark.skipif(
    tuple(int(x) for x in ort.__version__.split(".")[:2]) < (1, 23),
    reason="onnxruntime runs ONNX's Attention operator from 1.23",
)
@pytest.mark.parametrize("mask_shape", [[1, 1, 1, 5], [1, 2, 1, 5], [1, 1, 3, 5]])
def test_a_broadcast_mask_is_expanded_to_the_same_attention(mask_shape: list[int]) -> None:
    from noulxp.export.onnx_graph import expand_attention_masks

    rng = np.random.default_rng(0)
    q = rng.normal(size=(1, 2, 3, 4)).astype(np.float32)
    k = rng.normal(size=(1, 2, 5, 4)).astype(np.float32)
    v = rng.normal(size=(1, 2, 5, 4)).astype(np.float32)
    mask = np.ones(mask_shape, dtype=bool)
    mask[..., -2:] = False  # the last two keys are padding

    model = attention_model(mask_shape)
    assert expand_attention_masks(model) == 1
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    got = session.run(None, {"q": q, "k": k, "v": v, "m": mask})[0]
    assert np.allclose(got, reference(q, k, v, mask), atol=1e-5)


def test_fused_attention_is_refused_where_it_would_run_on_the_cpu() -> None:
    cuda = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    with pytest.raises(BackendUnavailable, match=r"1\.30"):
        check_fused_attention(23, cuda, "1.23.2")
    check_fused_attention(23, cuda, "1.30.0")
    check_fused_attention(23, ["CPUExecutionProvider"], "1.23.2")
    check_fused_attention(18, cuda, "1.23.2")


def test_the_runtime_reads_the_opset_from_the_manifest(toy_package, monkeypatch) -> None:  # noqa: F811
    """A package whose weights say opset 23 is refused on CUDA before any session exists."""
    import json

    from noulxp.profiles import encoder_markers
    from noulxp.runtime import load

    manifest = json.loads((toy_package / "noulxp.json").read_text())
    manifest["weights"]["opset"] = 23
    (toy_package / "noulxp.json").write_text(json.dumps(manifest))
    cuda = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    monkeypatch.setattr(encoder_markers, "choose_onnx_providers", lambda _provider: cuda)
    monkeypatch.setattr(ort, "__version__", "1.23.2")
    with pytest.raises(BackendUnavailable, match=r"opset 23"):
        load(toy_package, device="cuda")
