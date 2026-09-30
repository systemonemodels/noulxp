"""`opendxp export laya`: a Laya checkpoint (Laya itself or any Laya Studio fine-tune).

Laya is a ModernBERT (or mmBERT) encoder with a decision head that scores one
[MASK] marker per option, all questions in one pass. The graph is exported from
the `laya` package's own model (laya.common.build_model with the checkpoint's
weights, loaded exactly as laya.Agent loads them). template.json reproduces
laya.common.build_sequence and render_options; calibration.json holds the
temperatures of rl_agent_config.json as laya.Agent applies them, clamped to
[0.5, 5.0] (laya.common.clamp_temperature), with the per-option-count buckets.
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
from opendxp.profiles.encoder_markers import Template
from opendxp.spec import BASE
from opendxp.tokens import Tokens

TEMP_MIN, TEMP_MAX = 0.5, 5.0  # laya.common.TEMP_MIN / TEMP_MAX
QTYPE_ORDER = ("choice", "score", "noul")  # laya.common.QTYPES
BUCKETS = {"2": (2, 2), "3-5": (3, 5), "6-10": (6, 10), "11+": (11, None)}  # temp_bucket
MAX_OPTIONS, MAX_LEVELS = 20, 10  # the System One request limits


def clamp(value: Any) -> float:
    """laya.common.clamp_temperature."""
    try:
        t = float(value)
    except (TypeError, ValueError):
        return 1.0
    if t != t or t in (float("inf"), float("-inf")):
        return 1.0
    return min(TEMP_MAX, max(TEMP_MIN, t))


def template(cfg: dict[str, Any], special: dict[str, str]) -> dict[str, Any]:
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
            "choice": {"described": "{name}: {description}", "bare": "{name}"},
            "score": {"described": "level {index}: {description}"},
            "noul": {
                "false": {
                    "described": "false: {description}",
                    "bare": "false: no, the statement does not hold",
                },
                "true": {
                    "described": "true: {description}",
                    "bare": "true: yes, the statement holds",
                },
            },
        },
        "budgets": {
            "total": int(cfg.get("max_len", 512)),
            "head": int(cfg.get("head_max_len", 192)),
            "option": 48,
            "reserve": 16,
            "option_floor": 4,
            "head_floor": 8,
        },
        "truncation": "truncate-state",
        "reserved_text": "replace",
    }


def calibration(cfg: dict[str, Any]) -> dict[str, Any]:
    raw = list(cfg.get("temperature", [1.0, 1.0, 1.0]))
    by_type = {qtype: clamp(raw[i]) for i, qtype in enumerate(QTYPE_ORDER) if i < len(raw)}
    buckets, clamped = [], []
    for key, value in (cfg.get("temperature_by_options") or {}).items():
        qtype, size = key.split(":", 1)
        if qtype not in QTYPE_ORDER or size not in BUCKETS:
            raise ValueError(f"unknown temperature bucket {key!r}")
        low, high = BUCKETS[size]
        buckets.append({"type": qtype, "min": low, "max": high, "temperature": clamp(value)})
        if clamp(value) != float(value):
            clamped.append(f"{key}: {float(value):.6g} -> {clamp(value):.6g}")
    for i, value in enumerate(raw):
        if clamp(value) != float(value):
            clamped.append(f"temperature[{i}]: {float(value):.6g} -> {clamp(value):.6g}")
    order = {"choice": 0, "score": 1, "noul": 2}
    buckets.sort(key=lambda b: (order[b["type"]], b["min"]))
    return calibration_ok(
        {
            "standard": BASE,
            "temperature": by_type,
            "by_option_count": buckets,
            "source": {
                "file": "rl_agent_config.json",
                "temperature": raw,
                "temperature_by_options": cfg.get("temperature_by_options") or {},
                "rule": "laya applies every temperature clamped to [0.5, 5.0]",
                "clamped": clamped,
            },
        }
    )


def build_model(checkpoint: Path) -> Any:
    """The `laya` package's own model, loaded as laya.Agent loads it (float32, eval)."""
    require_transformers()
    from laya.common import build_model as laya_build_model
    from safetensors.torch import load_file

    cfg = json.loads((checkpoint / "rl_agent_config.json").read_text())
    model = laya_build_model(cfg, encoder_dir=str(checkpoint / "encoder"))
    model.load_state_dict(load_file(str(checkpoint / "model.safetensors")), strict=True)
    try:
        model.encoder.config.reference_compile = False
    except AttributeError:
        pass
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval()


class Graph:
    """Laya's forward returns (option logits, act logits); the graph keeps the first."""

    def __new__(cls, model: Any) -> Any:
        import torch

        class Wrapper(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = model

            def forward(
                self, input_ids, attention_mask, marker_positions, marker_mask, question_type
            ):  # type: ignore[no-untyped-def]
                logits, _ = self.model(
                    input_ids, attention_mask, marker_positions, marker_mask, question_type
                )
                return logits

        return Wrapper().eval()


def export(
    checkpoint: Path,
    out_dir: Path,
    *,
    name: str | None = None,
    weights_mode: str = "link",
    source: dict[str, Any] | None = None,
    opset: int = 18,
    log: Any = print,
) -> dict[str, Any]:
    checkpoint, out_dir = Path(checkpoint), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((checkpoint / "rl_agent_config.json").read_text())
    tok_cfg = tokenizer_config(checkpoint)
    special = {
        "cls": tok_cfg["cls_token"],
        "sep": tok_cfg["sep_token"],
        "marker": tok_cfg["mask_token"],
        "pad": tok_cfg["pad_token"],
    }
    tmpl = template(cfg, special)
    write_json(out_dir / "template.json", tmpl)
    write_json(out_dir / "calibration.json", calibration(cfg))
    how = {
        "tokenizer": place(tokenizer_path(checkpoint), out_dir / "tokenizer.json", weights_mode),
        "weights": place(checkpoint / "model.safetensors", out_dir / WEIGHTS_NAME, weights_mode),
    }
    log(f"placed tokenizer ({how['tokenizer']}) and weights ({how['weights']})")

    model = build_model(checkpoint)
    tokens = Tokens(out_dir / "tokenizer.json")
    batches = sample_batches(Template(tmpl), tokens)
    log("exporting the graph (torch.export) ...")
    graph = export_graph(
        Graph(model),
        trace_batch(batches),
        out_dir,
        out_dir / WEIGHTS_NAME,
        opset=opset,
        metadata={"odxp.source": name or str(cfg.get("model_name", "laya"))},
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
        name=name or f"laya:{cfg.get('model_name', 'checkpoint')}",
        profile="encoder-markers",
        files=files,
        limits={
            "max_tokens": tmpl["budgets"]["total"],
            "max_options": MAX_OPTIONS,
            "max_levels": MAX_LEVELS,
            "max_option_tokens": 48,
        },
        confidence={"choice": "entropy", "score": "entropy", "noul": "max-probability"},
        source={
            "model": "convaiinnovations/laya",
            "license": "Apache-2.0",
            "encoder": cfg.get("encoder"),
            # The folder's name, not where it sits: packages get published.
            "checkpoint": Path(checkpoint).name,
            "export": {"graph": graph, "verification": verification},
            **(source or {}),
        },
    )
    write_manifest(out_dir, data)
    return data
