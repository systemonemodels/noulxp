"""Exporting an encoder-markers decision model to ONNX with the standard signature.

The graph is exported with torch.export (the dynamo exporter), so batch, tokens
and options stay symbolic. Its weights are not copied: every parameter that is
a tensor of the checkpoint's model.safetensors becomes ONNX external data
pointing at that tensor's byte range, and the package carries the checkpoint's
weights file unchanged. Tensors the checkpoint stores in float16 or bfloat16
are declared in that type and cast to float32 in the graph, which is exactly
what the model's own runtime does when it loads them into a float32 model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from opendxp.export.common import safetensors_index
from opendxp.profiles.encoder_markers import INPUTS, OUTPUT
from opendxp.spec import BASE

WEIGHTS_NAME = "model.safetensors"
GRAPH_NAME = "model.onnx"
# An initializer this large that is not a checkpoint tensor would bloat the graph file.
INLINE_LIMIT_BYTES = 1 << 20


def export_graph(
    module: Any,
    sample: dict[str, Any],
    out_dir: Path,
    weights: Path,
    *,
    prefix: str = "model.",
    opset: int = 18,
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Export `module` (inputs INPUTS, output OUTPUT) to out_dir/model.onnx.

    `weights` is the checkpoint's safetensors file; `prefix` is what the module's
    parameter names add in front of the checkpoint's tensor names. The graph
    references WEIGHTS_NAME, which the caller places next to it.
    """
    import onnx_ir as ir
    import torch
    from onnxscript.optimizer import optimize_ir

    # PyTorch's fused TransformerEncoderLayer kernel has no ONNX translation.
    torch.backends.mha.set_fastpath_enabled(False)
    # Names rather than torch.export.Dim objects: the exporter makes each dimension dynamic
    # and keeps its name in the graph, with no declared range to check. A named Dim's range
    # includes 1, which torch 2.8 refuses for a model that treats a dimension of 1 specially.
    dynamic = {
        "input_ids": {0: "batch", 1: "tokens"},
        "attention_mask": {0: "batch", 1: "tokens"},
        "marker_positions": {0: "batch", 1: "options"},
        "marker_mask": {0: "batch", 1: "options"},
        "question_type": {0: "batch"},
    }
    args = tuple(torch.as_tensor(sample[name]) for name in INPUTS)
    with torch.no_grad():
        program = torch.onnx.export(
            module,
            args,
            dynamo=True,
            dynamic_shapes=dynamic,
            input_names=list(INPUTS),
            output_names=[OUTPUT],
            opset_version=opset,
            optimize=False,
            verify=False,
        )
    model = program.model

    header, data_start = safetensors_index(weights)

    def checkpoint_name(value: Any) -> str | None:
        if value.name is None or not value.name.startswith(prefix):
            return None
        name = value.name[len(prefix) :]
        info = header.get(name)
        if info is None or list(info["shape"]) != [int(d) for d in value.const_value.shape]:
            return None
        return name

    protected = {v.name for v in model.graph.initializers.values() if checkpoint_name(v)}

    def should_fold(node: Any) -> bool | None:
        # Folding a Transpose of a weight would write a transposed copy into the graph.
        if any(v is not None and v.name in protected for v in node.inputs):
            return False
        return None

    optimize_ir(model, should_fold=should_fold)

    dtypes = {
        "F32": ir.DataType.FLOAT,
        "F16": ir.DataType.FLOAT16,
        "BF16": ir.DataType.BFLOAT16,
        "I64": ir.DataType.INT64,
    }
    graph = model.graph
    first = next(iter(graph))
    casts = []
    mapped = cast = 0
    for value in list(graph.initializers.values()):
        name = checkpoint_name(value)
        if name is None:
            continue
        info = header[name]
        dtype = dtypes.get(info["dtype"])
        if dtype is None:
            continue
        start, end = info["data_offsets"]
        external = ir.ExternalTensor(
            WEIGHTS_NAME,
            data_start + start,
            end - start,
            dtype,
            shape=ir.Shape(info["shape"]),
            name=name,
            base_dir=str(out_dir),
        )
        if dtype == value.const_value.dtype:
            value.const_value = external
            mapped += 1
            continue
        if value.const_value.dtype != ir.DataType.FLOAT or dtype not in (
            ir.DataType.FLOAT16,
            ir.DataType.BFLOAT16,
        ):
            continue
        # The checkpoint stores it in half precision: read it as such, cast to float32.
        stored = ir.val(name + "::stored", dtype, list(info["shape"]), const_value=external)
        graph.initializers.pop(value.name)
        graph.register_initializer(stored)
        node = ir.node("Cast", [stored], {"to": ir.DataType.FLOAT}, name=name + "::cast")
        ir.convenience.replace_all_uses_with(value, node.outputs[0])
        node.outputs[0].name = value.name
        casts.append(node)
        mapped += 1
        cast += 1
    if casts:
        graph.insert_before(first, casts)

    inline = []
    for value in graph.initializers.values():
        tensor = value.const_value
        if isinstance(tensor, ir.ExternalTensor):
            continue
        if tensor is not None and tensor.nbytes > INLINE_LIMIT_BYTES:
            inline.append((value.name, tensor.nbytes))
    if inline:
        raise RuntimeError(
            f"{len(inline)} large tensors are not checkpoint tensors (e.g. {inline[0]}); "
            "the graph would copy weights"
        )
    model.producer_name = "opendxp"
    model.metadata_props.update(
        {"odxp.standard": BASE, "odxp.profile": "encoder-markers", **(metadata or {})}
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / GRAPH_NAME
    ir.save(model, str(path))
    attention = sum(1 for n in graph if n.op_type == "Attention" and n.domain in ("", "ai.onnx"))
    expanded = 0
    if attention:
        import onnx

        proto = onnx.load(str(path), load_external_data=False)
        expanded = expand_attention_masks(proto)
        if expanded:
            onnx.save(proto, str(path))
    return {
        "opset": opset,
        "initializers_from_checkpoint": mapped,
        "cast_from_half_precision": cast,
        "nodes": sum(1 for _ in graph),
        "attention_nodes": attention,
        "attention_masks_expanded": expanded,
        "graph_bytes": path.stat().st_size,
    }


def expand_attention_masks(model: Any) -> int:
    """Give each Attention node's mask its full (..., query length, key length) shape.

    ONNX lets attn_mask broadcast, but onnxruntime's kernels (1.23 to 1.30) require its
    second-to-last dimension to be the query length: a padding mask of shape
    (batch, 1, 1, keys) is refused. Each mask is expanded before its node; the result is
    the same attention. Returns how many masks were expanded.
    """
    from onnx import helper

    graph = model.graph
    nodes, changed = [], 0
    for node in graph.node:
        if node.op_type == "Attention" and node.domain in ("", "ai.onnx"):
            if len(node.input) > 3 and node.input[3]:
                q, k, mask = node.input[0], node.input[1], node.input[3]
                p = f"odxp_mask{changed}"
                nodes += [
                    helper.make_node("Shape", [mask], [p + "_shape"], start=0, end=-2),
                    helper.make_node("Shape", [q], [p + "_q"], start=-2, end=-1),
                    helper.make_node("Shape", [k], [p + "_k"], start=-2, end=-1),
                    helper.make_node(
                        "Concat", [p + "_shape", p + "_q", p + "_k"], [p + "_to"], axis=0
                    ),
                    helper.make_node("Expand", [mask, p + "_to"], [p + "_full"]),
                ]
                node.input[3] = p + "_full"
                changed += 1
        nodes.append(node)
    if changed:
        del graph.node[:]
        graph.node.extend(nodes)
    return changed


def compare_with_torch(
    onnx_path: Path, module: Any, batches: list[dict[str, Any]], provider: str = "cpu"
) -> dict[str, Any]:
    """Run the exported graph and the torch module on the same batches.

    Returns the largest logit difference and the largest difference between the
    softmax distributions over the valid options (temperature 1).
    """
    import numpy as np
    import torch

    from opendxp.providers import ort_session

    session, _ = ort_session(onnx_path, provider=provider)
    worst_logit = worst_p = 0.0
    same = total = 0
    for feeds in batches:
        got = session.run([OUTPUT], {k: np.asarray(v) for k, v in feeds.items()})[0]
        with torch.no_grad():
            want = module(*(torch.as_tensor(feeds[n]) for n in INPUTS)).float().numpy()
        mask = np.asarray(feeds["marker_mask"])
        for i in range(mask.shape[0]):
            k = int(mask[i].sum())
            a, b = got[i, :k].astype(np.float64), want[i, :k].astype(np.float64)
            worst_logit = max(worst_logit, float(np.abs(a - b).max()))
            pa, pb = np.exp(a - a.max()), np.exp(b - b.max())
            pa, pb = pa / pa.sum(), pb / pb.sum()
            worst_p = max(worst_p, float(np.abs(pa - pb).max()))
            same += int(pa.argmax() == pb.argmax())
            total += 1
    return {"rows": total, "same_argmax": same, "max_abs_logit": worst_logit, "max_abs_p": worst_p}
