"""`noulxp quantize`: derived packages at another precision, on toy models.

The encoder graph has the standard signature and the parts quantisation must
tell apart: embeddings, a Linear weight stored transposed, a float16 tensor cast
to float32, an attention-like product of activations, rotary angles made by a
product with a constant (Cos and Sin) and a decision head that reads
question_type. The GGUF path runs against a stand-in for llama.cpp that writes
a GGUF header of the asked-for type: what is tested is the package around it.
"""

from __future__ import annotations

import ctypes
import json
import struct
import sys
import types
from pathlib import Path
from typing import Any, ClassVar

import pytest

from noulxp.conformance import check, generate, read_jsonl
from noulxp.errors import NoulXPError
from noulxp.export import laya as laya_export
from noulxp.export.common import entry, manifest, place, write_json, write_manifest
from noulxp.package import open_package, sha256_file
from noulxp.validate import validate_package

D, VOCAB = 16, 512
LAYA_CFG = {
    "max_len": 96,
    "head_max_len": 40,
    "temperature": [1.0, 1.0, 1.0],
    "temperature_by_options": {},
}
SPECIAL = {"cls": "[CLS]", "sep": "[SEP]", "marker": "[MASK]", "pad": "[PAD]"}
REQUESTS = [
    {
        "id": "a",
        "request": {
            "state": "the customer wants a refund for the order",
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": "what does the customer want",
                    "criteria": {"refund": None, "cancel": "order", "track": "delivery"},
                },
                "angry": {"type": "noul", "instructions": "is the customer angry"},
                "level": {
                    "type": "score",
                    "instructions": "how urgent",
                    "criteria": ["low", "medium", "high"],
                },
            },
        },
    },
    {
        "id": "b",
        "request": {
            "state": "login password access help " * 6,
            "questions": {
                "team": {
                    "type": "choice",
                    "instructions": "which team",
                    "criteria": ["billing", "shipping", "access", "other"],
                },
            },
        },
    },
]


def toy_encoder(path: Path, opset: int = 18) -> None:
    """A small encoder-markers graph, its weights stored next to it as external data."""
    np = pytest.importorskip("numpy")
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(7)

    def weight(name: str, shape: tuple[int, ...], dtype: Any = np.float32) -> Any:
        return numpy_helper.from_array((rng.standard_normal(shape) * 0.3).astype(dtype), name)

    inits = [
        weight("embeddings", (VOCAB, D)),
        weight("w1_transposed", (D, D)),
        weight("w2_half", (D, D), np.float16),
        weight("type_embeddings", (3, D)),
        weight("head_weight", (D, D)),
        numpy_helper.from_array(
            (1.0 / (100.0 ** (np.arange(D) / D))).astype(np.float32)[None, :], "inv_freq"
        ),
        numpy_helper.from_array(np.array(0, np.int64), "zero"),
        numpy_helper.from_array(np.array(1, np.int64), "one"),
        numpy_helper.from_array(np.array([1], np.int64), "axis1"),
        numpy_helper.from_array(np.array([-1], np.int64), "axis_last"),
        numpy_helper.from_array(np.array([D], np.int64), "width"),
        numpy_helper.from_array(np.array([-1], np.int64), "minus_one"),
        numpy_helper.from_array(np.array(-1e4, np.float32), "floor"),
    ]
    n = helper.make_node
    nodes = [
        n("Gather", ["embeddings", "input_ids"], ["h0"], name="embed"),
        # rotary angles: positions times a constant frequency row, through a MatMul
        n("Shape", ["input_ids"], ["ids_shape"], name="ids_shape"),
        n("Gather", ["ids_shape", "one"], ["length"], name="length", axis=0),
        n("Range", ["zero", "length", "one"], ["positions"], name="positions"),
        n("Cast", ["positions"], ["positions_f"], name="positions_f", to=TensorProto.FLOAT),
        n("Unsqueeze", ["positions_f", "axis_last"], ["positions_col"], name="positions_col"),
        n("MatMul", ["positions_col", "inv_freq"], ["angles"], name="angles"),
        n("Cos", ["angles"], ["cos"], name="cos"),
        n("Sin", ["angles"], ["sin"], name="sin"),
        n("Mul", ["h0", "cos"], ["h_cos"], name="h_cos"),
        n("Mul", ["h0", "sin"], ["h_sin"], name="h_sin"),
        n("Add", ["h_cos", "h_sin"], ["h1_flat"], name="h1_flat"),
        # the batch size taken from question_type's shape, as exported graphs do: a shape,
        # not a value, so the encoder stays the encoder
        n("Shape", ["question_type"], ["qt_shape"], name="qt_shape"),
        n("Concat", ["qt_shape", "minus_one", "width"], ["h1_shape"], name="h1_shape", axis=0),
        n("Reshape", ["h1_flat", "h1_shape"], ["h1"], name="h1"),
        # a Linear stored transposed, as the converters keep checkpoint tensors
        n("Transpose", ["w1_transposed"], ["w1"], name="w1", perm=[1, 0]),
        n("MatMul", ["h1", "w1"], ["h2"], name="linear1"),
        # a float16 checkpoint tensor cast to float32 in the graph
        n("Cast", ["w2_half"], ["w2"], name="w2", to=TensorProto.FLOAT),
        n("MatMul", ["h2", "w2"], ["h3"], name="linear2"),
        # attention-like: a product of two activations
        n("Transpose", ["h3"], ["h3_t"], name="h3_t", perm=[0, 2, 1]),
        n("MatMul", ["h3", "h3_t"], ["scores"], name="scores"),
        n("Softmax", ["scores"], ["probs"], name="probs", axis=-1),
        n("MatMul", ["probs", "h3"], ["h4"], name="attend"),
        n("Add", ["h4", "h1"], ["h5"], name="residual"),
        # the decision head: everything from question_type on
        n("Gather", ["type_embeddings", "question_type"], ["t"], name="type_embed"),
        n("Unsqueeze", ["t", "axis1"], ["t3"], name="type_unsqueeze"),
        n("Add", ["h5", "t3"], ["h6"], name="head_add"),
        n("MatMul", ["h6", "head_weight"], ["h7"], name="head_linear"),
        n("Unsqueeze", ["marker_positions", "axis_last"], ["idx"], name="idx"),
        n("Shape", ["marker_positions"], ["mp_shape"], name="mp_shape"),
        n("Concat", ["mp_shape", "width"], ["idx_shape"], name="idx_shape", axis=0),
        n("Expand", ["idx", "idx_shape"], ["idx_full"], name="idx_full"),
        n("GatherElements", ["h7", "idx_full"], ["picked"], name="picked", axis=1),
        n("ReduceSum", ["picked", "axis_last"], ["logits"], name="logits", keepdims=0),
        n("Where", ["marker_mask", "logits", "floor"], ["option_logits"], name="masked"),
    ]
    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "tokens"]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "tokens"]),
        helper.make_tensor_value_info("marker_positions", TensorProto.INT64, ["batch", "options"]),
        helper.make_tensor_value_info("marker_mask", TensorProto.BOOL, ["batch", "options"]),
        helper.make_tensor_value_info("question_type", TensorProto.INT64, ["batch"]),
    ]
    output = helper.make_tensor_value_info("option_logits", TensorProto.FLOAT, ["batch", "options"])
    graph = helper.make_graph(nodes, "toy-encoder", inputs, [output], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.safetensors",
        size_threshold=1024,
    )


class RuntimeNative:
    """The float32 toy package's own answers stand in for a model's own code."""

    provenance: ClassVar[dict[str, str]] = {"runtime": "toy (the float32 package)"}

    def __init__(self, package: Path) -> None:
        from noulxp.runtime import load

        self.model = load(package, device="cpu")

    def predict(self, state: Any, questions: Any) -> dict[str, Any]:
        result: dict[str, Any] = self.model.predict(state, questions)
        return result


@pytest.fixture()
def encoder_package(tmp_path: Path, tokenizer_file: Path) -> Path:
    pytest.importorskip("onnxruntime")
    out = tmp_path / "toy-f32"
    out.mkdir()
    toy_encoder(out / "model.onnx")
    place(tokenizer_file, out / "tokenizer.json", "copy")
    write_json(out / "template.json", laya_export.template(LAYA_CFG, SPECIAL))
    write_json(out / "calibration.json", laya_export.calibration(LAYA_CFG))
    files = {
        "weights": entry(
            out, "model.onnx", format="onnx", opset=18, data=[entry(out, "model.safetensors")]
        ),
        "tokenizer": entry(out, "tokenizer.json"),
        "template": entry(out, "template.json"),
        "calibration": entry(out, "calibration.json"),
    }
    write_manifest(
        out,
        manifest(
            name="test/toy",
            profile="encoder-markers",
            files=files,
            limits={"max_tokens": 96, "max_options": 20},
            confidence={"choice": "entropy"},
            source={"model": "toy", "license": "Apache-2.0"},
        ),
    )
    native = RuntimeNative(out)
    generate(out, native, REQUESTS, log=lambda *_: None)
    native.model.close()
    return out


def problems(package: Path) -> list[str]:
    """validate_package's findings, but for the toy request set's coverage (two cases)."""
    return [p for p in validate_package(package) if not p.startswith("conformance coverage")]


def _graph(path: Path) -> Any:
    import onnx

    return onnx.load(str(path), load_external_data=False)


def _initializer_types(model: Any) -> dict[str, int]:
    return {t.name: t.data_type for t in model.graph.initializer}


def test_fp16_converts_the_encoder_products_and_keeps_head_and_angles(
    encoder_package: Path, tmp_path: Path
):
    from onnx import TensorProto

    from noulxp.quantize import quantize

    out = tmp_path / "toy-fp16"
    made = quantize(encoder_package, out, "fp16", log=lambda *_: None)
    model = _graph(out / "model.onnx")
    nodes = {n.name: n for n in model.graph.node}
    kinds = _initializer_types(model)

    # The encoder's two Linear products read float16 weights (the transposed one folded).
    assert kinds[nodes["linear1"].input[1]] == TensorProto.FLOAT16
    assert kinds[nodes["linear2"].input[1]] == TensorProto.FLOAT16
    assert "w1" not in nodes and "w2" not in nodes  # the weight transforms are folded
    # The attention-like products take cast activations and hand float32 back.
    assert nodes["scores"].output[0].endswith("::f16")
    assert any(n.op_type == "Cast" and n.output[0] == "scores" for n in model.graph.node)
    # The head and the rotary angles stay float32.
    assert kinds[nodes["head_linear"].input[1]] == TensorProto.FLOAT
    assert nodes["head_linear"].input[0] == "h6"
    assert list(nodes["angles"].input) == ["positions_col", "inv_freq"]
    assert kinds["inv_freq"] == TensorProto.FLOAT

    derived = made["source"]["derived"]
    assert derived["precision"] == "fp16"
    assert derived["products_float16"] == 4  # linear1, linear2, scores, attend
    assert derived["activation_casts"] == 3  # h1, probs and h3_t; h3 is linear2's own
    # type_embed, type_unsqueeze, head_add, head_linear, picked, logits, masked
    assert derived["float32_nodes"] == {"head": 7, "rotary": 8}
    assert derived["from"]["name"] == "test/toy"
    assert made["name"] == "test/toy:fp16"
    assert made["weights"]["data"][0]["path"] == "model.onnx.data"
    assert (
        made["conformance"]
        == json.loads((encoder_package / "noulxp.json").read_text())["conformance"]
    )
    assert sha256_file(out / "conformance.jsonl") == sha256_file(
        encoder_package / "conformance.jsonl"
    )
    assert not (out / "model.safetensors").exists()
    assert problems(out) == []

    report = check(out, device="cpu")
    assert report["passed"], report["failures"]
    assert report["max_abs_dp"] < 0.01
    assert report["argmax_agreement"]["agree"] == report["argmax_agreement"]["total"]


def test_hard_linked_weights_are_read(encoder_package: Path, tmp_path: Path):
    """The converters hard-link a checkpoint's weights into the package; onnx.load refuses
    such a file, so the quantiser reads the external data itself."""
    import os

    from noulxp.quantize import quantize

    os.link(encoder_package / "model.safetensors", tmp_path / "checkpoint.safetensors")
    made = quantize(encoder_package, tmp_path / "linked-fp16", "fp16", log=lambda *_: None)
    assert made["source"]["derived"]["products_float16"] == 4


def test_a_failed_run_leaves_no_package(encoder_package: Path, tmp_path: Path, monkeypatch):
    import noulxp.quantize as q

    def broken(*_: Any) -> Any:
        raise RuntimeError("no")

    monkeypatch.setattr(q, "half_products", broken)
    with pytest.raises(RuntimeError):
        q.quantize(encoder_package, tmp_path / "broken", "fp16", log=lambda *_: None)
    assert not (tmp_path / "broken").exists() and not (tmp_path / "broken.partial").exists()


def test_int8_quantizes_the_encoder_weights_and_not_the_head(encoder_package: Path, tmp_path: Path):
    pytest.importorskip("onnxruntime.quantization")
    from noulxp.quantize import quantize

    out = tmp_path / "toy-int8"
    made = quantize(encoder_package, out, "int8", log=lambda *_: None)
    model = _graph(out / "model.onnx")
    ops = [n.op_type for n in model.graph.node]
    # The two constant-weight products of the encoder, and nothing else.
    assert ops.count("MatMulInteger") == 2
    names = {n.name for n in model.graph.node}
    assert {"head_linear", "angles", "scores", "attend"} <= names  # still float MatMuls
    derived = made["source"]["derived"]
    assert derived["products_int8"] == 2
    assert derived["excluded_head_and_rotary"] == 2  # head_linear and angles
    assert problems(out) == []

    report = check(out, device="cpu")
    assert report["cases"] == len(REQUESTS)
    assert report["max_abs_dp"] < 0.1  # int8 is checked, not assumed: here it is close


def test_quantize_refuses_what_it_cannot_do(encoder_package: Path, tmp_path: Path):
    from noulxp.quantize import quantize

    with pytest.raises(NoulXPError, match="encoder-markers"):
        quantize(encoder_package, tmp_path / "a", "q8_0", log=lambda *_: None)
    with pytest.raises(NoulXPError, match="one of"):
        quantize(encoder_package, tmp_path / "b", "int4", log=lambda *_: None)
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "x").write_text("x")
    with pytest.raises(NoulXPError, match="not empty"):
        quantize(encoder_package, busy, "fp16", log=lambda *_: None)

    bare = tmp_path / "bare"
    bare.mkdir()
    data = json.loads((encoder_package / "noulxp.json").read_text())
    data.pop("conformance")
    for key in ("tokenizer", "template", "calibration"):
        place(encoder_package / data[key]["path"], bare / data[key]["path"], "copy")
    for name in ("model.onnx", "model.safetensors"):
        place(encoder_package / name, bare / name, "copy")
    (bare / "noulxp.json").write_text(json.dumps(data))
    with pytest.raises(NoulXPError, match="no conformance file"):
        quantize(bare, tmp_path / "c", "fp16", log=lambda *_: None)


def test_the_cli_writes_a_derived_package(encoder_package: Path, tmp_path: Path, capsys):
    from noulxp.cli import main

    out = tmp_path / "cli-fp16"
    assert main(["quantize", str(encoder_package), str(out), "--to", "fp16", "--copy"]) == 0
    assert "fp16" in capsys.readouterr().out
    assert open_package(out).name == "test/toy:fp16"


# ---- causal-letters: GGUF ---------------------------------------------------------------

GGUF_STRING, GGUF_ARRAY, GGUF_UINT32, GGUF_FLOAT32 = 8, 9, 4, 6


def write_gguf(path: Path, file_type: int) -> None:
    """A GGUF header with a few metadata keys (arrays first, as real files have) and no tensors."""

    def string(text: str) -> bytes:
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw

    kv = [
        string("general.architecture") + struct.pack("<I", GGUF_STRING) + string("toy"),
        string("tokenizer.ggml.tokens")
        + struct.pack("<IIQ", GGUF_ARRAY, GGUF_STRING, 3)
        + b"".join(string(t) for t in ("a", "b", "c")),
        string("tokenizer.ggml.scores")
        + struct.pack("<IIQ", GGUF_ARRAY, GGUF_FLOAT32, 3)
        + struct.pack("<3f", 0.0, -1.0, -2.0),
        string("general.file_type") + struct.pack("<II", GGUF_UINT32, file_type),
    ]
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)) + b"".join(kv))


def test_gguf_metadata_reads_past_arrays(tmp_path: Path):
    from noulxp.quantize import file_type_name, gguf_file_type, gguf_metadata

    path = tmp_path / "x.gguf"
    write_gguf(path, 32)
    assert gguf_file_type(path) == 32 and file_type_name(32) == "BF16"
    found = gguf_metadata(path, {"general.architecture", "tokenizer.ggml.tokens"})
    assert found == {"general.architecture": "toy", "tokenizer.ggml.tokens": ["a", "b", "c"]}


class Params(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, Any]]] = [
        ("nthread", ctypes.c_int32),
        ("ftype", ctypes.c_int),
        ("allow_requantize", ctypes.c_bool),
    ]


@pytest.fixture()
def fake_llama(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """llama_cpp as far as quantize uses it: the call is recorded, a header is written."""
    calls: list[dict[str, Any]] = []

    def llama_model_quantize(src: bytes, dst: bytes, ref: Any) -> int:
        params = ref._obj
        calls.append(
            {"src": src.decode(), "ftype": params.ftype, "allow": bool(params.allow_requantize)}
        )
        write_gguf(Path(dst.decode()), params.ftype)
        return 0

    fake = types.SimpleNamespace(
        LLAMA_FTYPE_ALL_F32=0,
        LLAMA_FTYPE_MOSTLY_F16=1,
        LLAMA_FTYPE_MOSTLY_Q8_0=7,
        LLAMA_FTYPE_MOSTLY_Q4_K_S=14,
        LLAMA_FTYPE_MOSTLY_Q4_K_M=15,
        LLAMA_FTYPE_MOSTLY_Q5_K_S=16,
        LLAMA_FTYPE_MOSTLY_Q5_K_M=17,
        LLAMA_FTYPE_MOSTLY_Q6_K=18,
        LLAMA_FTYPE_MOSTLY_BF16=32,
        llama_backend_init=lambda: None,
        llama_model_quantize_default_params=lambda: Params(0, 7, False),
        llama_model_quantize=llama_model_quantize,
    )
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)
    return calls


def causal_package(root: Path, tokenizer_file: Path, file_type: int) -> Path:
    from noulxp.export import decider

    checkpoint = root / f"ckpt-{file_type}"
    checkpoint.mkdir()
    place(tokenizer_file, checkpoint / "tokenizer.json", "copy")
    (checkpoint / "decider_config.json").write_text(
        json.dumps({"temperature": 1.1, "max_options": 12, "isolated_levels": True})
    )
    write_gguf(checkpoint / "model.gguf", file_type)
    out = root / f"pkg-{file_type}"
    decider.export(checkpoint, out, weights_mode="copy", log=lambda *_: None)
    case = {
        "id": "c1",
        "request": {"state": "x", "questions": {"q": {"type": "noul", "instructions": "yes"}}},
        "expected": {"q": {"type": "noul", "probabilities": {"false": 0.4, "true": 0.6}}},
    }
    (out / "conformance.jsonl").write_text(json.dumps(case) + "\n")
    data = json.loads((out / "noulxp.json").read_text())
    data["conformance"] = {**entry(out, "conformance.jsonl"), "cases": 1, "questions": 1}
    (out / "noulxp.json").write_text(json.dumps(data))
    return out


def test_gguf_is_quantized_by_llama_cpp_into_a_derived_package(
    tmp_path: Path, tokenizer_file: Path, fake_llama: list[dict[str, Any]]
):
    from noulxp.quantize import gguf_file_type, quantize

    source = causal_package(tmp_path, tokenizer_file, 32)  # BF16
    out = tmp_path / "q8"
    made = quantize(source, out, "q8_0", threads=3, log=lambda *_: None)
    assert fake_llama == [{"src": str(source / "model.gguf"), "ftype": 7, "allow": False}]
    assert gguf_file_type(out / "model.gguf") == 7
    assert made["weights"]["sha256"] == sha256_file(out / "model.gguf")
    derived = made["source"]["derived"]
    assert (derived["from_type"], derived["file_type"]) == ("BF16", "Q8_0")
    assert made["name"] == "mapika/decider:q8_0"
    for name in ("tokenizer.json", "prompt.json", "calibration.json", "conformance.jsonl"):
        assert sha256_file(out / name) == sha256_file(source / name)
    assert open_package(out).profile == "causal-letters"
    assert read_jsonl(out / "conformance.jsonl")[0]["id"] == "c1"


def test_gguf_refuses_to_requantize_unless_asked(
    tmp_path: Path, tokenizer_file: Path, fake_llama: list[dict[str, Any]]
):
    from noulxp.quantize import quantize

    q8 = causal_package(tmp_path, tokenizer_file, 7)
    with pytest.raises(NoulXPError, match="Q8_0 weights"):
        quantize(q8, tmp_path / "a", "q4_k_m", log=lambda *_: None)
    with pytest.raises(NoulXPError, match="already Q8_0"):
        quantize(q8, tmp_path / "b", "q8_0", allow_requantize=True, log=lambda *_: None)
    made = quantize(q8, tmp_path / "c", "q4_k_m", allow_requantize=True, log=lambda *_: None)
    assert fake_llama[-1]["allow"] is True
    assert made["source"]["derived"]["file_type"] == "Q4_K_M"
    with pytest.raises(NoulXPError, match="causal-letters package takes"):
        quantize(q8, tmp_path / "d", "fp16", log=lambda *_: None)
