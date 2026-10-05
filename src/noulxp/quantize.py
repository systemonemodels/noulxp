"""`noulxp quantize`: a derived package, the same model with its weights at another precision.

A derived package (SPEC.md section 10) keeps everything of its source package
but the weights: the tokenizer, template or prompt, calibration and, above all,
the conformance file, which was recorded by the model's own code on the
original weights. It is NoulXP compatible only if it passes that file, so
`noulxp check` on the result says whether the precision costs the model any of
its answers.

encoder-markers (ONNX):

    fp16  half-precision matrix products, the way PyTorch's autocast computes them:
          every MatMul and Gemm of the encoder (and ONNX's Attention, opset 23) reads
          float16 weights and float16 activations and hands float32 back. The rest
          stays float32: embeddings, normalisation, softmax and the residual stream,
          the decision head (every node that reads `question_type`) and the rotary
          angles (every node that leads to a Cos or a Sin), whose positions lose
          whole radians at float16.
    int8  dynamic int8 quantisation for CPUs (onnxruntime.quantization.quantize_dynamic):
          the encoder's weight matrices as int8 per output column, activations as
          uint8 per call (MatMulInteger); the decision head stays float32.

causal-letters (GGUF): llama.cpp's own quantiser (llama_model_quantize) from F32,
F16 or BF16 weights to f32, f16, bf16, q8_0, q6_k, q5_k_m, q5_k_s, q4_k_m or
q4_k_s. Re-quantising already quantised weights is refused unless asked for.
"""

from __future__ import annotations

import ctypes
import importlib.metadata
import json
import os
import shutil
import struct
import tempfile
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from noulxp import __version__
from noulxp.errors import BackendUnavailable, NoulXPError, PackageError
from noulxp.export.common import entry, now, place, write_manifest
from noulxp.package import Package, open_package

ONNX_TARGETS = ("fp16", "int8")
# llama.cpp's file types (llama.h, enum llama_ftype) by the names `--to` takes.
GGUF_TARGETS = {
    "f32": "ALL_F32",
    "f16": "MOSTLY_F16",
    "bf16": "MOSTLY_BF16",
    "q8_0": "MOSTLY_Q8_0",
    "q6_k": "MOSTLY_Q6_K",
    "q5_k_m": "MOSTLY_Q5_K_M",
    "q5_k_s": "MOSTLY_Q5_K_S",
    "q4_k_m": "MOSTLY_Q4_K_M",
    "q4_k_s": "MOSTLY_Q4_K_S",
}
TARGETS = (*ONNX_TARGETS, *GGUF_TARGETS)
# general.file_type values (llama_ftype) and their names, for the record and for the
# requantising check: F32, F16 and BF16 are weights as trained, the others are not.
FILE_TYPES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    32: "BF16",
}
UNQUANTIZED = {"F32", "F16", "BF16"}
WEIGHTS_GRAPH, WEIGHTS_DATA, WEIGHTS_GGUF = "model.onnx", "model.onnx.data", "model.gguf"
# The products that run at float16 under `fp16`, and ONNX's largest finite float16.
HALF_OPS = ("MatMul", "Gemm", "Attention")
HALF_MAX = 65504.0


def quantize(
    source: str | Path,
    out_dir: str | Path,
    to: str,
    *,
    name: str | None = None,
    weights_mode: str = "link",
    threads: int | None = None,
    allow_requantize: bool = False,
    log: Any = print,
) -> dict[str, Any]:
    """Write a derived package of `source` at precision `to` into `out_dir`; its manifest."""
    to = to.lower()
    if to not in TARGETS:
        raise NoulXPError(f"--to is one of {', '.join(TARGETS)}, not {to!r}")
    package = open_package(source)
    if package.profile == "encoder-markers" and to not in ONNX_TARGETS:
        raise NoulXPError(f"an encoder-markers package takes {' or '.join(ONNX_TARGETS)}")
    if package.profile == "causal-letters" and to not in GGUF_TARGETS:
        raise NoulXPError(f"a causal-letters package takes {', '.join(GGUF_TARGETS)}")
    problems = package.verify(hashes=True)
    if problems:
        raise PackageError("the source package does not verify: " + "; ".join(problems))
    conformance = package.manifest.get("conformance")
    if not isinstance(conformance, dict) or not conformance.get("path"):
        raise NoulXPError(
            "the source package has no conformance file: a derived package is checked against "
            "the answers its model's own code gave on the original weights (SPEC.md 10)"
        )
    final = Path(out_dir)
    if final.exists() and any(final.iterdir()):
        raise NoulXPError(f"{final} is not empty")
    # Written beside the target and renamed into place when whole: a failed run leaves
    # nothing that looks like a package.
    out = final.with_name(final.name + ".partial")
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    try:
        manifest = _derive(package, out, to, name, weights_mode, threads, allow_requantize, log)
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    if final.exists():
        final.rmdir()
    out.rename(final)
    return manifest


def _derive(
    package: Package,
    out: Path,
    to: str,
    name: str | None,
    weights_mode: str,
    threads: int | None,
    allow_requantize: bool,
    log: Any,
) -> dict[str, Any]:
    conformance = package.manifest["conformance"]
    manifest: dict[str, Any] = json.loads(json.dumps(package.manifest))
    for key in ("tokenizer", "template", "prompt", "calibration", "conformance"):
        value = manifest.get(key)
        if isinstance(value, dict) and isinstance(value.get("path"), str):
            place(package.resolve(value["path"]), out / value["path"], weights_mode)

    if package.profile == "encoder-markers":
        weights, details = _onnx(package, out, to, log)
    else:
        weights, details = _gguf(package, out, to, threads, allow_requantize, log)
    manifest["weights"] = weights
    manifest["name"] = name or f"{package.name}:{to}"
    source_info = dict(manifest.get("source") or {})
    source_info["derived"] = {
        "from": {
            "name": package.name,
            "weights_sha256": package.entry("weights").get("sha256"),
            "conformance_sha256": conformance.get("sha256"),
        },
        "precision": to,
        **details,
        "by": f"noulxp {__version__}",
        "at": now(),
    }
    manifest["source"] = source_info
    write_manifest(out, manifest)
    return manifest


# --- encoder-markers -------------------------------------------------------------------


def _require_onnx() -> Any:
    try:
        import onnx
    except ImportError as exc:
        raise BackendUnavailable(
            'noulxp quantize needs onnx: pip install "noulxp[export]"'
        ) from exc
    return onnx


def load_graph(package: Package) -> Any:
    """The package's ONNX graph with its weights read into it.

    onnx.load would read the external data itself, but it refuses a file with more than
    one hard link, and the converters hard-link a checkpoint's weights into the package
    (the published bytes, not a second copy). Each tensor is read here by its recorded
    offset and length, from files inside the package only.
    """
    import onnx
    from onnx import TensorProto

    model = onnx.load(str(package.file("weights")), load_external_data=False)
    tensors = list(model.graph.initializer)
    for node in model.graph.node:
        tensors += [a.t for a in node.attribute if a.type == onnx.AttributeProto.TENSOR]
    files: dict[Path, Any] = {}
    try:
        for tensor in tensors:
            if tensor.data_location != TensorProto.EXTERNAL:
                continue
            info = {e.key: e.value for e in tensor.external_data}
            path = package.resolve(info["location"])
            if path not in files:
                files[path] = open(path, "rb")  # closed below
            fh = files[path]
            fh.seek(int(info.get("offset") or 0))
            length = info.get("length")
            tensor.raw_data = fh.read(int(length)) if length is not None else fh.read()
            tensor.data_location = TensorProto.DEFAULT
            del tensor.external_data[:]
    finally:
        for fh in files.values():
            fh.close()
    return model


def _onnx(package: Package, out: Path, to: str, log: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    onnx = _require_onnx()
    model = load_graph(package)
    graph = model.graph
    name_nodes(graph)
    folded = fold_weight_transforms(graph)
    head = descendants(graph, {"question_type"})
    rotary = rotary_nodes(graph)
    keep = head | rotary
    log(
        f"folded {folded} weight transforms; float32 kept for {len(head)} head nodes "
        f"and {len(rotary)} rotary-angle nodes"
    )
    details: dict[str, Any] = {
        "weight_transforms_folded": folded,
        "float32_nodes": {"head": len(head), "rotary": len(rotary)},
        "onnx": onnx.__version__,
    }
    with tempfile.TemporaryDirectory(prefix="noulxp-quantize-") as tmp:
        if to == "fp16":
            types = value_types(model, Path(tmp))
            stats = half_products(graph, keep, types)
            log(f"float16 products: {stats}")
            details.update(
                method="float16 matrix products, float32 elsewhere (as torch.autocast)",
                **stats,
            )
            _save(model, out)
        else:
            stats = _int8(model, keep, Path(tmp), out)
            log(f"int8: {stats}")
            details.update(
                method="onnxruntime.quantization.quantize_dynamic: MatMul with constant weights, "
                "int8 per output column, uint8 activations",
                **stats,
            )
    graph_model = onnx.load(str(out / WEIGHTS_GRAPH), load_external_data=False)
    graph_model.metadata_props.add(key="noulxp.derived", value=to)
    onnx.save_model(graph_model, str(out / WEIGHTS_GRAPH))
    data = [entry(out, WEIGHTS_DATA)] if (out / WEIGHTS_DATA).is_file() else []
    weights = entry(out, WEIGHTS_GRAPH, format="onnx", opset=opset_of(graph_model))
    if data:
        weights["data"] = data
    return weights, details


def opset_of(model: Any) -> int:
    return max(int(op.version) for op in model.opset_import if op.domain in ("", "ai.onnx"))


def _save(model: Any, out: Path) -> None:
    import onnx

    onnx.save_model(
        model,
        str(out / WEIGHTS_GRAPH),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=WEIGHTS_DATA,
        size_threshold=1024,
    )


def name_nodes(graph: Any) -> None:
    """Give every node a unique name: quantize_dynamic and the keep-lists address nodes by it."""
    used: set[str] = set()
    for i, node in enumerate(graph.node):
        name = node.name or f"noulxp_{node.op_type}_{i}"
        candidate, n = name, 1
        while candidate in used:
            candidate, n = f"{name}_{n}", n + 1
        node.name = candidate
        used.add(candidate)


def fold_weight_transforms(graph: Any) -> int:
    """Replace each Cast or Transpose of an initializer by the initializer it computes.

    The converters keep a checkpoint's tensors as stored (a float16 tensor cast to
    float32 in the graph, a Linear weight transposed in the graph), so the package can
    point at the published weights. A derived package writes its own weights, and a
    quantiser has to see each MatMul's constant weight matrix as it is multiplied.
    """
    from onnx import helper, numpy_helper

    inputs = {i.name for i in graph.input}
    folded, changed = 0, True
    while changed:
        changed = False
        initializers = {t.name: t for t in graph.initializer}
        for node in list(graph.node):
            if node.op_type not in ("Cast", "Transpose") or node.domain not in ("", "ai.onnx"):
                continue
            if len(node.input) != 1 or node.input[0] not in initializers:
                continue
            if node.input[0] in inputs:
                continue
            array = numpy_helper.to_array(initializers[node.input[0]])
            attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            if node.op_type == "Transpose":
                perm = attrs.get("perm") or list(reversed(range(array.ndim)))
                result = array.transpose(perm).copy()
            else:
                result = array.astype(helper.tensor_dtype_to_np_dtype(attrs["to"]))
            graph.initializer.append(numpy_helper.from_array(result, node.output[0]))
            graph.node.remove(node)
            folded += 1
            changed = True
    drop_unused_initializers(graph)
    return folded


def drop_unused_initializers(graph: Any) -> None:
    used = {name for node in graph.node for name in node.input if name}
    used |= {o.name for o in graph.output}
    for tensor in list(graph.initializer):
        if tensor.name not in used:
            graph.initializer.remove(tensor)


def descendants(graph: Any, values: set[str]) -> set[str]:
    """The names of every node that reads, directly or not, one of `values`."""
    consumers: dict[str, list[Any]] = defaultdict(list)
    for node in graph.node:
        for name in node.input:
            if name:
                consumers[name].append(node)
    seen: set[str] = set()
    queue = deque(values)
    reached = set(values)
    while queue:
        for node in consumers.get(queue.popleft(), ()):
            if node.name in seen:
                continue
            seen.add(node.name)
            for name in node.output:
                if name and name not in reached:
                    reached.add(name)
                    queue.append(name)
    return seen


def rotary_nodes(graph: Any) -> set[str]:
    """Every Cos and Sin node and every node that leads to one: the rotary angles.

    A position times a frequency is an angle of up to the context length in radians;
    float16 holds 8,192 to within 4, so the angles (and the products that make them)
    stay float32, as transformers keeps them under autocast.
    """
    producers = {name: node for node in graph.node for name in node.output if name}
    start = [n for n in graph.node if n.op_type in ("Cos", "Sin")]
    seen = {n.name for n in start}
    queue = deque(start)
    while queue:
        for name in queue.popleft().input:
            node = producers.get(name)
            if node is not None and node.name not in seen:
                seen.add(node.name)
                queue.append(node)
    return seen


def value_types(model: Any, workdir: Path) -> dict[str, int]:
    """The element type of every value in the graph, from ONNX's shape inference.

    Run on a copy saved with its weights outside the graph, so a model over protobuf's
    2 GB limit is inferred too.
    """
    import onnx

    copy = onnx.ModelProto()
    copy.CopyFrom(model)
    path, inferred = workdir / "typed.onnx", workdir / "typed-inferred.onnx"
    onnx.save_model(
        copy,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="typed.onnx.data",
        size_threshold=1024,
    )
    del copy
    onnx.shape_inference.infer_shapes_path(str(path), str(inferred))
    typed = onnx.load(str(inferred), load_external_data=False)
    types: dict[str, int] = {}
    for info in [*typed.graph.input, *typed.graph.value_info, *typed.graph.output]:
        elem = info.type.tensor_type.elem_type
        if elem:
            types[info.name] = elem
    for tensor in typed.graph.initializer:
        types[tensor.name] = tensor.data_type
    return types


def half_products(graph: Any, keep: set[str], types: dict[str, int]) -> dict[str, int]:
    """Run the graph's matrix products at float16, everything else as it was.

    Each converted node reads float16 copies of its float32 weights and a Cast of
    its float32 activations, and its float32 output is a Cast of its float16 one.
    An activation cast once is reused by every converted node that reads it. An
    Attention node keeps its softmax at float32 (softmax_precision), as fused
    attention kernels do under autocast, and an additive float mask is floored at
    float16's lowest finite value before its cast.
    """
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    initializers = {t.name: t for t in graph.initializer}
    float_inputs = {"MatMul": (0, 1), "Gemm": (0, 1, 2), "Attention": (0, 1, 2, 3, 4, 5)}
    half_values: dict[str, str] = {}
    nodes: list[Any] = []
    converted = clamped = weights = casts = 0
    floor = "noulxp::f16_floor"

    def half_of(name: str, mask: bool = False) -> str:
        nonlocal clamped, weights, casts
        key = name + ("::mask" if mask else "")
        if key in half_values:
            return half_values[key]
        half = f"{key}::f16"
        if name in initializers:
            array = numpy_helper.to_array(initializers[name]).astype(np.float32)
            clamped += int(np.count_nonzero(np.abs(array) > HALF_MAX))
            values = np.clip(array, -HALF_MAX, HALF_MAX).astype(np.float16)
            graph.initializer.append(numpy_helper.from_array(values, half))
            weights += 1
        else:
            source = name
            if mask:
                if floor not in initializers:
                    tensor = numpy_helper.from_array(np.array(-HALF_MAX, np.float32), floor)
                    graph.initializer.append(tensor)
                    initializers[floor] = tensor
                source = f"{name}::floored"
                nodes.append(helper.make_node("Max", [name, floor], [source], name=source))
            nodes.append(
                helper.make_node(
                    "Cast", [source], [half], name=half + "::cast", to=TensorProto.FLOAT16
                )
            )
            casts += 1
        half_values[key] = half
        return half

    for node in list(graph.node):
        wanted = float_inputs.get(node.op_type)
        if wanted is None or node.domain not in ("", "ai.onnx") or node.name in keep:
            nodes.append(node)
            continue
        floats = [
            i
            for i in wanted
            if i < len(node.input)
            and node.input[i]
            and types.get(node.input[i]) == TensorProto.FLOAT
        ]
        result = node.output[0] if node.output else ""
        # Only the first output is cast back: a node using any other is left as it is.
        if not floats or types.get(result) != TensorProto.FLOAT or any(node.output[1:]):
            nodes.append(node)
            continue
        attention = node.op_type == "Attention"
        for i in floats:
            node.input[i] = half_of(node.input[i], mask=attention and i == 3)
        if attention and not any(a.name == "softmax_precision" for a in node.attribute):
            node.attribute.append(helper.make_attribute("softmax_precision", TensorProto.FLOAT))
        node.output[0] = f"{result}::f16"
        # A converted node that reads this result takes the float16 one: the float32 copy
        # is its exact widening, so nothing is rounded twice and no cast is added.
        half_values[result] = node.output[0]
        nodes.append(node)
        nodes.append(
            helper.make_node(
                "Cast", [node.output[0]], [result], name=result + "::f32", to=TensorProto.FLOAT
            )
        )
        converted += 1
    del graph.node[:]
    graph.node.extend(nodes)
    drop_unused_initializers(graph)
    return {
        "products_float16": converted,
        "weights_float16": weights,
        "activation_casts": casts,
        "weights_clamped": clamped,
    }


def _int8(model: Any, keep: set[str], tmp: Path, out: Path) -> dict[str, Any]:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise BackendUnavailable(
            'int8 quantisation needs onnxruntime: pip install "noulxp[onnx]"'
        ) from exc
    import onnx

    excluded = sorted(n.name for n in model.graph.node if n.name in keep and n.op_type == "MatMul")
    staged = tmp / "folded.onnx"
    onnx.save_model(
        model,
        str(staged),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="folded.onnx.data",
        size_threshold=1024,
    )
    quantize_dynamic(
        staged,
        out / WEIGHTS_GRAPH,
        op_types_to_quantize=["MatMul"],
        per_channel=True,
        reduce_range=False,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=excluded,
        use_external_data_format=True,
        extra_options={"MatMulConstBOnly": True},
    )
    result = onnx.load(str(out / WEIGHTS_GRAPH), load_external_data=False)
    ops = [n.op_type for n in result.graph.node]
    return {
        "products_int8": ops.count("MatMulInteger"),
        "products_float32": ops.count("MatMul"),
        "excluded_head_and_rotary": len(excluded),
        "onnxruntime": importlib.metadata.version(_ort_distribution()),
    }


def _ort_distribution() -> str:
    for dist in ("onnxruntime", "onnxruntime-gpu"):
        try:
            importlib.metadata.version(dist)
            return dist
        except importlib.metadata.PackageNotFoundError:
            continue
    return "onnxruntime"


# --- causal-letters --------------------------------------------------------------------


def _gguf(
    package: Package, out: Path, to: str, threads: int | None, allow_requantize: bool, log: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import llama_cpp as lc
    except ImportError as exc:
        raise BackendUnavailable(
            'quantising GGUF weights needs llama-cpp-python: pip install "noulxp[gguf]"'
        ) from exc
    source = package.file("weights")
    have = file_type_name(gguf_file_type(source))
    want = GGUF_TARGETS[to].split("_", 1)[1]
    if have == want:
        raise NoulXPError(f"{source.name} is already {have}")
    if have not in UNQUANTIZED and not allow_requantize:
        raise NoulXPError(
            f"{source.name} holds {have} weights: quantise from F32, F16 or BF16 weights "
            "(re-quantising loses more; --allow-requantize does it anyway)"
        )
    target = out / WEIGHTS_GGUF
    lc.llama_backend_init()
    params = lc.llama_model_quantize_default_params()
    params.ftype = getattr(lc, f"LLAMA_FTYPE_{GGUF_TARGETS[to]}")
    params.nthread = int(threads or os.cpu_count() or 4)
    params.allow_requantize = bool(allow_requantize)
    log(f"llama.cpp: {source.name} ({have}) -> {target.name} ({want})")
    code = lc.llama_model_quantize(str(source).encode(), str(target).encode(), ctypes.byref(params))
    if code != 0 or not target.is_file():
        raise NoulXPError(f"llama.cpp could not quantise {source.name} to {to} (code {code})")
    got = file_type_name(gguf_file_type(target))
    details = {
        "method": "llama.cpp llama_model_quantize",
        "from_type": have,
        "file_type": got,
        "llama_cpp_python": _version("llama_cpp_python"),
    }
    return entry(out, WEIGHTS_GGUF, format="gguf"), details


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def file_type_name(value: int | None) -> str:
    if value is None:
        return "unknown"
    return FILE_TYPES.get(value, f"type {value}")


# GGUF metadata value types: their struct formats (scalars), 8 = string, 9 = array.
_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?"}
_SCALARS |= {10: "<Q", 11: "<q", 12: "<d"}


def gguf_metadata(path: str | Path, keys: set[str]) -> dict[str, Any]:
    """The values of `keys` in a GGUF file's metadata (versions 2 and 3), read without llama.cpp."""
    found: dict[str, Any] = {}
    with open(path, "rb") as fh:
        if fh.read(4) != b"GGUF":
            raise PackageError(f"{Path(path).name} is not a GGUF file")
        (version,) = struct.unpack("<I", fh.read(4))
        if version < 2:
            raise PackageError(f"{Path(path).name} is GGUF version {version}; 2 or 3 is read")
        _tensors, count = struct.unpack("<QQ", fh.read(16))
        for _ in range(count):
            key = _string(fh)
            (kind,) = struct.unpack("<I", fh.read(4))
            wanted = key in keys
            value = _value(fh, kind, wanted)
            if wanted:
                found[key] = value
                if len(found) == len(keys):
                    break
    return found


def gguf_file_type(path: str | Path) -> int | None:
    value = gguf_metadata(path, {"general.file_type"}).get("general.file_type")
    return int(value) if value is not None else None


def _string(fh: Any) -> str:
    (length,) = struct.unpack("<Q", fh.read(8))
    return fh.read(length).decode("utf-8", errors="replace")


def _value(fh: Any, kind: int, keep: bool) -> Any:
    if kind in _SCALARS:
        fmt = _SCALARS[kind]
        return struct.unpack(fmt, fh.read(struct.calcsize(fmt)))[0]
    if kind == 8:
        return _string(fh)
    if kind == 9:
        (item, length) = struct.unpack("<IQ", fh.read(12))
        if not keep and item in _SCALARS:
            fh.seek(length * struct.calcsize(_SCALARS[item]), os.SEEK_CUR)
            return None
        items = [_value(fh, item, keep) for _ in range(length)]
        return items if keep else None
    raise PackageError(f"unknown GGUF metadata type {kind}")
