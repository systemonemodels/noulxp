"""Julia 1 (Supersonic Labs): its own inference, for writing conformance files.

An mmBERT-small encoder with a marker-scoring head. The model repository ships
its own `julia` package, which pins transformers 5.0; this is its inference
path rebuilt without it, from SupersonicLabs/Julia-1 at revision
a85b127321d580d65176c89ced8273f305745d85 (Apache-2.0):

    julia/model.py          JuliaDecisionModel (inference subset) and _selected_head
    julia/data.py           sequence(): marker serialisation, budgets, strict checks
    julia/router/engine.py  _pack(): right padding to a multiple of 8
    julia/typed.py          the softmax and the answer format

Checked against the authors' 100 published parity cases: the same answer on
all 100, logits within 1.1e-4, and identical under transformers 5.0 and 5.17.
Tokenisation goes through `tokenizers` directly (identical ids on 641 test
strings). The lengths are the published inference policy's (8,192 tokens, a
512-token head). One adapter is ours: System One criteria become Julia rows
(Julia's own API refuses list choices and null descriptions; here an option's
name is its text).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from opendxp.errors import PackageError

QTYPES = {"choice": 0, "score": 1, "noul": 2}
OPTION_TOKENS = 48  # the per-option token contract (julia/data.py:84-86)


def _read_json(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _model_class() -> Any:
    import torch
    from torch import nn
    from torch.nn import functional as F

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

    class JuliaDecisionModel(nn.Module):  # type: ignore[misc]
        """The inference subset of julia/model.py; parameter names match model.safetensors."""

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
            # In the checkpoint, never used for inference.
            self.act_head = nn.Sequential(
                nn.Linear(width + 4, 256), nn.GELU(), nn.Linear(256, n_act)
            )
            self.register_buffer("temperature", torch.ones(3))

        def forward(
            self, input_ids: Any, attention_mask: Any, marker_pos: Any, marker_mask: Any, qtype: Any
        ) -> Any:
            hidden = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            hidden = hidden + self.type_emb(qtype)[:, None, :]
            padding = ~attention_mask.bool()
            last = len(self.head.layers) - 1
            for i, layer in enumerate(self.head.layers):
                if i == last:
                    hidden = selected_head(layer, hidden, marker_pos, attention_mask)
                else:
                    hidden = layer(hidden, src_key_padding_mask=padding)
            scores = self.scorer(hidden).squeeze(-1).float()
            return scores.masked_fill(~marker_mask, -1e4)

    return JuliaDecisionModel


class _Tokens:
    """`tokenizer(text, add_special_tokens=False)["input_ids"]`, from tokenizer.json alone."""

    def __init__(self, directory: Path) -> None:
        from tokenizers import Tokenizer

        self.backend = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.backend.no_truncation()
        self.backend.no_padding()
        config = _read_json(directory / "tokenizer_config.json")
        ident = self.backend.token_to_id
        self.mask_token = config["mask_token"]
        self.mask_token_id = ident(config["mask_token"])
        self.cls_token_id = ident(config["cls_token"])
        self.sep_token_id = ident(config["sep_token"])
        self.pad_token_id = ident(config["pad_token"])
        if None in (self.mask_token_id, self.cls_token_id, self.sep_token_id, self.pad_token_id):
            raise PackageError("the tokenizer does not define mask, cls, sep and pad tokens")

    def __call__(self, text: str) -> list[int]:
        ids: list[int] = self.backend.encode(text, add_special_tokens=False).ids
        return ids


def _sequence(
    tok: _Tokens, row: dict[str, Any], max_length: int, head_length: int
) -> dict[str, Any]:
    """julia/data.py:72-110, strict: the same order, budgets and refusals."""
    state = row["state"]
    if any(tok.mask_token in text for text in [state, row["question"], *row["options"]]):
        raise ValueError("the request contains the model's reserved marker token")

    def clean(text: str) -> str:
        return text.replace(tok.mask_token, " ")

    head = tok(f"{row['type']} question: {clean(row['question'])}")
    option_ids = [tok(" " + clean(x)) for x in row["options"]]
    if any(len(x) > OPTION_TOKENS for x in option_ids):
        raise ValueError(f"an option is longer than the model's {OPTION_TOKENS}-token limit")
    options = [[tok.mask_token_id] + x[:OPTION_TOKENS] for x in option_ids]
    budget = head_length - sum(map(len, options))
    if budget < 16:
        per_option = max(4, (head_length - 16) // len(options))
        options = [x[:per_option] for x in options]
        budget = head_length - sum(map(len, options))
    if len(head) > budget or any(
        len(x) != len(y) + 1 for x, y in zip(options, option_ids, strict=True)
    ):
        raise ValueError("the question and options are too long for the model; shorten them")
    ids = [tok.cls_token_id, *head[: max(8, budget)], tok.sep_token_id]
    markers = []
    for option in options:
        markers.append(len(ids))
        ids.extend(option)
    ids.append(tok.sep_token_id)
    state_ids = tok(clean(state))
    room = max_length - len(ids) - 1
    if room < 1 or len(state_ids) > room:
        raise ValueError(f"the state is too long: this model reads up to {max_length} tokens")
    return {
        "ids": ids + state_ids + [tok.sep_token_id],
        "markers": markers,
        "qtype": QTYPES[row["type"]],
    }


def _rows(
    state: str, questions: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[tuple[str, str, list[str], list[str]]]]:
    """System One questions as Julia rows (julia/typed.py:6-32), with the adapter above."""
    rows, meta = [], []
    for qid, q in questions.items():
        kind, criteria = q.get("type"), q.get("criteria")
        if kind == "choice":
            if isinstance(criteria, list):
                keys = labels = [str(x) for x in criteria]
            elif isinstance(criteria, dict):
                keys = list(criteria)
                labels = [v if v else k for k, v in criteria.items()]
            else:
                raise ValueError("a choice question needs options")
        elif kind == "score":
            if not isinstance(criteria, list):
                raise ValueError("a score question needs its levels, lowest first")
            labels, keys = [str(x) for x in criteria], [str(i) for i in range(len(criteria))]
        elif kind == "noul":
            keys = ["false", "true"]
            given = criteria or {}
            if not isinstance(given, dict) or not set(given) <= set(keys):
                raise ValueError('a noul question takes only "true" and "false" descriptions')
            labels = [given.get(k) or k for k in keys]
        else:
            raise ValueError(f"unsupported question type {kind!r}")
        if not 2 <= len(labels) <= 20:
            raise ValueError("a question needs 2 to 20 options")
        rows.append(
            {
                "state": state,
                "question": str(q.get("instructions", "")),
                "type": kind,
                "options": labels,
            }
        )
        meta.append((qid, kind, keys, labels))
    return rows, meta


class Julia:
    """Julia 1 on the CPU, answering in the System One format."""

    def __init__(self, checkpoint: Path, threads: int = 4) -> None:
        import torch
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel

        config = _read_json(checkpoint / "julia_config.json")
        found = (
            config.get("format_version"),
            config.get("architecture"),
            config.get("weight_dtype"),
        )
        if found != (1, "JuliaDecisionModel", "float32"):
            raise PackageError(f"not a Julia 1 checkpoint: {found}")
        policy_path = checkpoint / "inference-policy.json"
        policy = _read_json(policy_path) if policy_path.is_file() else {}
        self.max_length = int(policy.get("max_length", 8192))
        self.head_length = int(policy.get("head_length", 512))
        try:
            from transformers.initialization import no_init_weights
        except ImportError:  # older transformers
            from transformers.modeling_utils import no_init_weights
        torch.set_num_threads(threads)
        encoder_config = AutoConfig.from_pretrained(checkpoint / "encoder")
        encoder_config.reference_compile = False
        with no_init_weights():
            encoder = AutoModel.from_config(encoder_config, attn_implementation="sdpa")
            model = _model_class()(
                encoder, int(config["head_layers"]), int(config["n_act"]), float(config["dropout"])
            )
        state = load_file(str(checkpoint / "model.safetensors"))
        model.load_state_dict(state, strict=True, assign=True)
        self.model = model.eval()
        folder = checkpoint / "tokenizer"
        self.tokens = _Tokens(folder if folder.is_dir() else checkpoint)
        self.torch = torch

    def _pack(self, encoded: list[dict[str, Any]]) -> dict[str, Any]:
        torch = self.torch
        length = min(self.max_length, (max(len(x["ids"]) for x in encoded) + 7) // 8 * 8)
        count = max(len(x["markers"]) for x in encoded)
        n = len(encoded)
        ids = torch.full((n, length), self.tokens.pad_token_id, dtype=torch.long)
        attention = torch.zeros((n, length), dtype=torch.long)
        marker_pos = torch.zeros((n, count), dtype=torch.long)
        marker_mask = torch.zeros((n, count), dtype=torch.bool)
        for i, item in enumerate(encoded):
            ids[i, : len(item["ids"])] = torch.tensor(item["ids"])
            attention[i, : len(item["ids"])] = 1
            marker_pos[i, : len(item["markers"])] = torch.tensor(item["markers"])
            marker_mask[i, : len(item["markers"])] = True
        qtype = torch.tensor([item["qtype"] for item in encoded], dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": attention,
            "marker_pos": marker_pos,
            "marker_mask": marker_mask,
            "qtype": qtype,
        }

    def system_one(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        rows, meta = _rows(state, questions)
        encoded = [_sequence(self.tokens, row, self.max_length, self.head_length) for row in rows]
        with self.torch.inference_mode():
            values = self.model(**self._pack(encoded)).float()
        answers: dict[str, Any] = {}
        for i, (qid, kind, keys, labels) in enumerate(meta):
            z = values[i, : len(encoded[i]["markers"])].tolist()
            if not all(math.isfinite(x) for x in z):
                raise FloatingPointError("the model returned non-finite scores")
            top = max(z)
            e = [math.exp(x - top) for x in z]
            total = sum(e)
            p = [x / total for x in e]
            best = max(range(len(p)), key=p.__getitem__)
            if kind == "choice":
                answer = {
                    "type": "choice",
                    "choice": keys[best],
                    "probabilities": {k: round(x, 4) for k, x in zip(keys, p, strict=True)},
                    "confidence": round(p[best], 4),
                }
            elif kind == "score":
                answer = {
                    "type": "score",
                    "score": round(sum(j * x for j, x in enumerate(p)), 4),
                    "legend": dict(zip(keys, labels, strict=True)),
                    "probabilities": {k: round(x, 4) for k, x in zip(keys, p, strict=True)},
                    "confidence": round(p[best], 4),
                }
            else:
                answer = {"type": "noul", "noul": round(p[1], 4)}
            answers[qid] = answer
        tokens = sum(len(x["ids"]) for x in encoded)
        return {"answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0}}
