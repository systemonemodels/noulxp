"""`opendxp export julia`: Julia 1 (Supersonic Labs) as an encoder-markers package.

Julia 1 is an mmBERT-small encoder with a decision head that scores one <mask>
marker per option. The graph is exported from the inference subset of
julia/model.py (SupersonicLabs/Julia-1 at a85b127321d580d65176c89ced8273f305745d85,
Apache-2.0), with the last head layer computed at the markers only, exactly as
the package's CPU engine runs it. template.json reproduces julia/data.py
sequence() with strict encoding, the model's published inference policy
(inference-policy.json: max_length 8192, head_length 512, strict), and
calibration is temperature 1: Julia publishes no calibration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from opendxp.export.common import (
    entry,
    manifest,
    place,
    require_transformers,
    write_json,
    write_manifest,
)
from opendxp.export.encoder import (
    calibration_ok,
    sample_batches,
    tokenizer_config,
    tokenizer_path,
    trace_batch,
)
from opendxp.export.onnx_graph import GRAPH_NAME, WEIGHTS_NAME, compare_with_torch, export_graph
from opendxp.package import sha256_file
from opendxp.profiles.encoder_markers import Template
from opendxp.spec import BASE
from opendxp.tokens import Tokens

MAX_OPTIONS = 20


def template(special: dict[str, str], total: int = 8192, head: int = 512) -> dict[str, Any]:
    return {
        "standard": BASE,
        "profile": "encoder-markers",
        "special_tokens": special,
        "question_type_ids": {"choice": 0, "score": 1, "noul": 2},
        "layout": [
            {"token": "cls"},
            {"part": "head"},
            {"token": "sep"},
            {"part": "options"},
            {"token": "sep"},
            {"part": "state"},
            {"token": "sep"},
        ],
        "head": "{type} question: {instructions}",
        "options": {
            "prefix": " ",
            # julia/typed.py: an option is its description; our adapter uses the name when
            # there is none (the engine's policy for System One requests).
            "choice": {"described": "{description}", "bare": "{name}"},
            "score": {"described": "{description}"},
            "noul": {
                "false": {"described": "{description}", "bare": "false"},
                "true": {"described": "{description}", "bare": "true"},
            },
        },
        "budgets": {
            "total": total,
            "head": head,
            "option": 48,
            "reserve": 16,
            "option_floor": 4,
            "head_floor": 8,
        },
        "truncation": "strict",
        "reserved_text": "reject",
    }


def build_model(checkpoint: Path) -> Any:
    """JuliaDecisionModel (inference subset), weights loaded strictly, float32, eval."""
    require_transformers()
    import torch
    from safetensors.torch import load_file
    from torch import nn
    from torch.nn import functional as F
    from transformers import AutoConfig, AutoModel

    config = json.loads((checkpoint / "julia_config.json").read_text())
    found = (config.get("format_version"), config.get("architecture"), config.get("weight_dtype"))
    if found != (1, "JuliaDecisionModel", "float32"):
        raise ValueError(f"not a Julia 1 checkpoint: {found}")

    def selected_head(layer: Any, hidden: Any, selected: Any, attention_mask: Any) -> Any:
        """julia/model.py:76-94: the last pre-norm block, computed only at the markers."""
        width = hidden.shape[-1]
        positions = selected[:, :, None].expand(-1, -1, width)
        normalized = layer.norm1(hidden)
        attn = layer.self_attn
        query = F.linear(
            normalized.gather(1, positions), attn.in_proj_weight[:width], attn.in_proj_bias[:width]
        )
        key, value = F.linear(
            normalized, attn.in_proj_weight[width:], attn.in_proj_bias[width:]
        ).chunk(2, dim=-1)
        batch, heads = hidden.shape[0], attn.num_heads

        def split(x: Any) -> Any:
            return x.reshape(batch, -1, heads, width // heads).transpose(1, 2)

        attended = F.scaled_dot_product_attention(
            split(query),
            split(key),
            split(value),
            attn_mask=attention_mask[:, None, None, :].bool(),
            dropout_p=0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, selected.shape[1], width)
        result = hidden.gather(1, positions) + attn.out_proj(attended)
        return result + layer.linear2(layer.activation(layer.linear1(layer.norm2(result))))

    class JuliaDecisionModel(nn.Module):
        def __init__(self, encoder: Any, head_layers: int, n_act: int, dropout: float) -> None:
            super().__init__()
            self.encoder = encoder
            width = encoder.config.hidden_size
            layer = nn.TransformerEncoderLayer(
                width, max(1, width // 64), 4 * width, dropout, batch_first=True, norm_first=True
            )
            self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False)
            self.type_emb = nn.Embedding(3, width)
            self.scorer = nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1)
            )
            self.act_head = nn.Sequential(
                nn.Linear(width + 4, 256), nn.GELU(), nn.Linear(256, n_act)
            )
            self.register_buffer("temperature", torch.ones(3))

        def forward(self, input_ids, attention_mask, marker_positions, marker_mask, question_type):  # type: ignore[no-untyped-def]
            hidden = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            hidden = hidden + self.type_emb(question_type)[:, None, :]
            padding = ~attention_mask.bool()
            last = len(self.head.layers) - 1
            for i, layer in enumerate(self.head.layers):
                if i == last:
                    hidden = selected_head(layer, hidden, marker_positions, attention_mask)
                else:
                    hidden = layer(hidden, src_key_padding_mask=padding)
            scores = self.scorer(hidden).squeeze(-1).float()
            return scores.masked_fill(~marker_mask, -1e4)

    try:
        from transformers.initialization import no_init_weights
    except ImportError:  # older transformers
        from transformers.modeling_utils import no_init_weights
    encoder_config = AutoConfig.from_pretrained(checkpoint / "encoder")
    encoder_config.reference_compile = False
    with no_init_weights():
        encoder = AutoModel.from_config(encoder_config, attn_implementation="sdpa")
        model = JuliaDecisionModel(
            encoder, int(config["head_layers"]), int(config["n_act"]), float(config["dropout"])
        )
    model.load_state_dict(load_file(str(checkpoint / "model.safetensors")), strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval()


class Graph:
    """The module the exporter traces: the model under the name `model`."""

    def __new__(cls, model: Any) -> Any:
        import torch

        class Wrapper(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = model

            def forward(
                self, input_ids, attention_mask, marker_positions, marker_mask, question_type
            ):  # type: ignore[no-untyped-def]
                return self.model(
                    input_ids, attention_mask, marker_positions, marker_mask, question_type
                )

        return Wrapper().eval()


OFFICIAL_NAMES = {
    "marker_pos": "marker_positions",
    "qtype": "question_type",
    "logits": "option_logits",
}
FOLDED_FROM = "pkg.onnxscript.optimizer.folded_from"


def map_official_graph(graph_path: Path, out_dir: Path, weights: Path) -> dict[str, Any]:
    """SupersonicLabs/Julia-1-ONNX's own graph, mapped to the OpenDXP signature.

    Its inputs differ only in name (marker_pos, qtype, logits). Its weights live
    in model.onnx.data, but they are the checkpoint's tensors: each initializer
    named after a parameter is pointed at that tensor in model.safetensors, and
    each one the exporter folded (a transposed weight, recorded in its
    folded_from metadata) is rebuilt as a Transpose of its checkpoint tensor.
    The 577 MB data file is never needed.
    """
    import ast

    import onnx_ir as ir

    from opendxp.export.common import safetensors_index

    model = ir.load(str(graph_path))
    graph = model.graph
    renamed = {}
    for value in [*graph.inputs, *graph.outputs]:
        if value.name in OFFICIAL_NAMES:
            renamed[value.name] = OFFICIAL_NAMES[value.name]
            value.name = OFFICIAL_NAMES[value.name]
    header, data_start = safetensors_index(weights)

    def checkpoint_tensor(key: str) -> Any:
        info = header[key]
        start, end = info["data_offsets"]
        return ir.ExternalTensor(
            WEIGHTS_NAME,
            data_start + start,
            end - start,
            ir.DataType.FLOAT,
            shape=ir.Shape(info["shape"]),
            name=key,
            base_dir=str(out_dir),
        )

    first = next(iter(graph))
    nodes, sources = [], {}
    direct = transposed = 0
    for name, value in list(graph.initializers.items()):
        tensor = value.const_value
        shape = [int(d) for d in tensor.shape]
        key = name.removeprefix("model.")
        if key in header and header[key]["shape"] == shape and header[key]["dtype"] == "F32":
            value.const_value = checkpoint_tensor(key)
            direct += 1
            continue
        origin = ast.literal_eval(value.metadata_props.get(FOLDED_FROM, "[]"))
        if len(origin) == 1 and origin[0].removeprefix("model.") in header:
            key = origin[0].removeprefix("model.")
            if header[key]["shape"][::-1] == shape and header[key]["dtype"] == "F32":
                if key not in sources:
                    sources[key] = ir.val(
                        f"{key}::checkpoint",
                        ir.DataType.FLOAT,
                        header[key]["shape"],
                        const_value=checkpoint_tensor(key),
                    )
                    graph.register_initializer(sources[key])
                node = ir.node("Transpose", [sources[key]], {"perm": [1, 0]}, name=f"{name}::t")
                graph.initializers.pop(name)
                ir.convenience.replace_all_uses_with(value, node.outputs[0])
                node.outputs[0].name = name
                nodes.append(node)
                transposed += 1
                continue
        if isinstance(tensor, ir.ExternalTensor):
            raise ValueError(f"{name} is neither a checkpoint tensor nor a transposed one")
    graph.insert_before(first, nodes)
    model.metadata_props.update(
        {
            "odxp.standard": BASE,
            "odxp.profile": "encoder-markers",
            "odxp.source": "SupersonicLabs/Julia-1-ONNX model.onnx, inputs renamed",
        }
    )
    ir.save(model, str(out_dir / GRAPH_NAME))
    return {
        "graph": "SupersonicLabs/Julia-1-ONNX/model.onnx",
        "graph_sha256": sha256_file(graph_path),
        "renamed": renamed,
        "initializers_from_checkpoint": direct,
        "transposes_rebuilt_from_checkpoint": transposed,
        "opset": 18,
        "graph_bytes": (out_dir / GRAPH_NAME).stat().st_size,
    }


def export(
    checkpoint: Path,
    out_dir: Path,
    *,
    name: str = "supersonic-labs/julia-1",
    weights_mode: str = "link",
    max_tokens: int | None = None,
    head_tokens: int = 512,
    official_onnx: Path | None = None,
    source: dict[str, Any] | None = None,
    log: Any = print,
) -> dict[str, Any]:
    checkpoint, out_dir = Path(checkpoint), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder_config = json.loads((checkpoint / "encoder" / "config.json").read_text())
    total = int(max_tokens or encoder_config.get("max_position_embeddings", 8192))
    tok_cfg = tokenizer_config(checkpoint)
    special = {
        "cls": tok_cfg["cls_token"],
        "sep": tok_cfg["sep_token"],
        "marker": tok_cfg["mask_token"],
        "pad": tok_cfg["pad_token"],
    }
    tmpl = template(special, total=total, head=head_tokens)
    write_json(out_dir / "template.json", tmpl)
    calibration = calibration_ok(
        {
            "standard": BASE,
            "temperature": {"choice": 1.0, "score": 1.0, "noul": 1.0},
            "source": "Julia 1 publishes no calibration (inference-policy.json: calibration null); "
            "the temperature buffer in the checkpoint is [1, 1, 1] and is never read.",
        }
    )
    write_json(out_dir / "calibration.json", calibration)
    how = {
        "tokenizer": place(tokenizer_path(checkpoint), out_dir / "tokenizer.json", weights_mode),
        "weights": place(checkpoint / "model.safetensors", out_dir / WEIGHTS_NAME, weights_mode),
    }
    log(f"placed tokenizer ({how['tokenizer']}) and weights ({how['weights']})")

    model = build_model(checkpoint)
    tokens = Tokens(out_dir / "tokenizer.json")
    batches = sample_batches(Template(tmpl), tokens)
    if official_onnx is not None:
        log(f"mapping the official graph {official_onnx} to the OpenDXP signature ...")
        graph = map_official_graph(Path(official_onnx), out_dir, out_dir / WEIGHTS_NAME)
    else:
        log("exporting the graph (torch.export) ...")
        graph = export_graph(
            Graph(model),
            trace_batch(batches),
            out_dir,
            out_dir / WEIGHTS_NAME,
            metadata={"odxp.source": name},
        )
    log(f"graph written: {graph}")
    verification = compare_with_torch(out_dir / GRAPH_NAME, Graph(model), batches)
    log(f"onnxruntime vs torch: {verification}")

    files = {
        "weights": entry(
            out_dir,
            GRAPH_NAME,
            format="onnx",
            opset=graph["opset"],
            data=[entry(out_dir, WEIGHTS_NAME)],
        ),
        "tokenizer": entry(out_dir, "tokenizer.json"),
        "template": entry(out_dir, "template.json"),
        "calibration": entry(out_dir, "calibration.json"),
    }
    data = manifest(
        name=name,
        profile="encoder-markers",
        files=files,
        limits={"max_tokens": total, "max_options": MAX_OPTIONS, "max_option_tokens": 48},
        confidence={"choice": "max-probability", "score": "max-probability"},
        source={
            "model": "SupersonicLabs/Julia-1",
            "license": "Apache-2.0",
            # The folder's name, not where it sits: packages get published.
            "checkpoint": Path(checkpoint).name,
            "export": {"graph": graph, "verification": verification},
            **(source or {}),
        },
    )
    write_manifest(out_dir, data)
    return data
