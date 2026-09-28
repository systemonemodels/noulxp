"""Mapping an existing ONNX export onto the OpenDXP signature and the checkpoint's weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from opendxp.export.julia import FOLDED_FROM, map_official_graph


def _official_style_graph(path: Path, w: np.ndarray, b: np.ndarray) -> None:
    """A graph like Julia-1-ONNX's: old input names, one weight folded to its transpose."""
    ir = pytest.importorskip("onnx_ir")
    ids = ir.val("input_ids", ir.DataType.INT64, ["batch", "tokens"])
    attention = ir.val("attention_mask", ir.DataType.INT64, ["batch", "tokens"])
    positions = ir.val("marker_pos", ir.DataType.INT64, ["batch", "options"])
    mask = ir.val("marker_mask", ir.DataType.BOOL, ["batch", "options"])
    qtype = ir.val("qtype", ir.DataType.INT64, ["batch"])
    folded = ir.val("val_7", const_value=ir.tensor(w.T.copy(), name="val_7"))
    folded.metadata_props[FOLDED_FROM] = "['model.scorer.weight']"
    bias = ir.val("model.scorer.bias", const_value=ir.tensor(b, name="model.scorer.bias"))
    minus = ir.val("floor", const_value=ir.tensor(np.array(-1e4, np.float32), name="floor"))
    axis = ir.val("axis", const_value=ir.tensor(np.array([-1], np.int64), name="axis"))
    nodes = [
        ir.node("GatherElements", [ids, positions], {"axis": 1}, num_outputs=1),
    ]
    token = nodes[-1].outputs[0]
    nodes.append(ir.node("Cast", [token], {"to": ir.DataType.FLOAT}))
    nodes.append(ir.node("Unsqueeze", [nodes[-1].outputs[0], axis]))
    nodes.append(ir.node("MatMul", [nodes[-1].outputs[0], folded]))
    nodes.append(ir.node("ReduceSum", [nodes[-1].outputs[0], axis], {"keepdims": 0}))
    nodes.append(ir.node("Add", [nodes[-1].outputs[0], bias]))
    nodes.append(ir.node("Where", [mask, nodes[-1].outputs[0], minus]))
    out = nodes[-1].outputs[0]
    out.name = "logits"
    out.type = ir.TensorType(ir.DataType.FLOAT)
    graph = ir.Graph(
        [ids, attention, positions, mask, qtype],
        [out],
        nodes=nodes,
        initializers=[folded, bias, minus, axis],
        opset_imports={"": 18},
        name="official",
    )
    ir.save(ir.Model(graph, ir_version=10), str(path))


def test_official_graph_is_renamed_and_reads_the_checkpoint(tmp_path: Path) -> None:
    ort = pytest.importorskip("onnxruntime")
    st = pytest.importorskip("safetensors.numpy")
    rng = np.random.default_rng(0)
    w = rng.standard_normal((3, 1)).astype(np.float32)  # a Linear weight [out, in]
    b = rng.standard_normal((1,)).astype(np.float32)
    original = tmp_path / "official.onnx"
    _official_style_graph(original, w, b)
    out = tmp_path / "package"
    out.mkdir()
    st.save_file(
        {"scorer.weight": w, "scorer.bias": b, "unused": np.zeros(4, np.float32)},
        str(out / "model.safetensors"),
    )

    report = map_official_graph(original, out, out / "model.safetensors")
    assert report["renamed"] == {
        "marker_pos": "marker_positions",
        "qtype": "question_type",
        "logits": "option_logits",
    }
    assert report["initializers_from_checkpoint"] == 1
    assert report["transposes_rebuilt_from_checkpoint"] == 1

    feeds = {
        "input_ids": np.array([[5, 7, 9, 11]], np.int64),
        "attention_mask": np.ones((1, 4), np.int64),
        "marker_mask": np.array([[True, True, False]]),
    }
    before = ort.InferenceSession(str(original)).run(
        None,
        {**feeds, "marker_pos": np.array([[0, 2, 0]], np.int64), "qtype": np.array([0], np.int64)},
    )[0]
    mapped = ort.InferenceSession(str(out / "model.onnx"))
    names = sorted(i.name for i in mapped.get_inputs())
    assert names == sorted(
        ["input_ids", "attention_mask", "marker_positions", "marker_mask", "question_type"]
    )
    after = mapped.run(
        ["option_logits"],
        {
            **feeds,
            "marker_positions": np.array([[0, 2, 0]], np.int64),
            "question_type": np.array([0], np.int64),
        },
    )[0]
    np.testing.assert_allclose(after, before, rtol=0, atol=1e-6)
