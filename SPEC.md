# OpenDXP 0.2

**The Open Decision Exchange Protocol: a portable standard for calibrated single-pass decision models.**

Status: version 0.2. It adds the HTTP and MCP bindings (sections 11 and 12)
and, to the causal-letters profile, typed layouts, chat templates and
rotations (6.7, 6.8); every 0.1 package is a 0.2 package. Features new in 0.2
are marked (0.2). Reference implementation: the `opendxp` package in this
repository. Licence: Apache-2.0.

The key words MUST, MUST NOT, SHOULD, SHOULD NOT and MAY are used as in RFC 2119.
The JSON Schemas in `schemas/` are normative for the structure of every file;
this document is normative for their meaning.

## 1. Scope

A **System One model** answers typed questions about a state in one forward
pass per question (or per question row), with a probability for every option.
Every such model published so far ships its own inference code. OpenDXP defines
a **package** that any OpenDXP **engine** can run without code written for that
model, and a **conformance** method that proves the engine runs it faithfully.

OpenDXP has two **profiles**, one per architecture family:

| Profile | Architecture | Weights | Declarative input | Answer read from |
| --- | --- | --- | --- | --- |
| `encoder-markers` | bidirectional encoder, one marker token per option | ONNX | `template.json` | one logit per marker |
| `causal-letters` | decoder language model, lettered options | GGUF | `prompt.json` | the letter tokens' logits at the answer slot |

A model that fits neither profile can still be listed; it is not OpenDXP
compatible and runs only with its own code.

OpenDXP is a protocol in four parts, so that any application can ask any
decision model the same way, on any machine:

1. **Requests and answers** (section 3): one format for a state and typed
   questions, and for the calibrated answers.
2. **Packages** (sections 4 to 8): a model as data that any engine runs.
3. **Conformance** (sections 9 and 10): proof that an engine answers as the
   model's own code does.
4. **Bindings** (sections 11 and 12): how requests travel. Over HTTP between
   an application and a server, and over the Model Context Protocol between an
   AI agent and a tool.

Terms used below:

- **Native runtime**: the model's own code (for example the `laya` package).
- **Reference runtime**: the runtime in this repository, which reads only the package.
- **Engine**: any program that runs OpenDXP packages. The reference runtime is one.
- **Row**: one forward pass. An encoder-markers question is one row; a
  causal-letters question is one row, one row per level (section 6.4) or one
  row per rotation of its options (section 6.8).

## 2. Nothing in a package is executed

A package is data: a manifest, weights in ONNX or GGUF, a `tokenizer.json`,
declarative JSON files and a conformance file. An engine MUST NOT execute
anything a package contains other than the weights graph through its ONNX or
GGUF runtime. Templates are filled by placeholder substitution only (section
3.5); there is no template language, no Python, no Jinja. A chat template
(6.7) is stored as the text it renders to, with `{system}` and `{user}` in
place of the two messages; its Jinja source is not part of the package.

## 3. Requests and answers

### 3.1 Request

```json
{"state": "text the model reads",
 "questions": {"<id>": {"type": "choice" | "score" | "noul",
                        "instructions": "text",
                        "criteria": ...}}}
```

| Type | `criteria` | Option keys, in order |
| --- | --- | --- |
| `choice` | `{name: description or null, ...}` or `[name, ...]`, at least 2, names unique and non-empty | the names |
| `score` | `[level description, ...]`, at least 2, lowest level first | `"0"`, `"1"`, ... |
| `noul` | absent, `null`, or an object with only `"false"` and/or `"true"` | `"false"`, `"true"` |

`instructions` MUST be a string when present; absent means the empty string.
The package's `limits.max_options` (and `max_levels` for score) bound the
option count; an engine MUST refuse a request over them.

### 3.2 State

A string state is used as it is. A JSON state (object or array) is rendered
with `json.dumps(state, ensure_ascii=False)` semantics: separators `", "` and
`": "`, keys in insertion order, non-ASCII kept. A causal-letters prompt MAY
declare, in its `state` block:

- `index_arrays_from: N`: arrays of N or more items are annotated before
  rendering, each item becoming `{"_index": i, ...item}` (an object) or
  `{"_index": i, "value": item}`;
- `json: "indent-2"` (0.2): JSON is rendered with two-space indents,
  separators `","` and `": "` (`json.dumps(state, indent=2, ensure_ascii=False)`);
- `messages: "role-content"` (0.2): a conversation, a non-empty array of
  objects that each have a string `role` and a string `content`, is rendered
  as one `role: content` line per turn, joined by newlines;
- `empty` (0.2): the text used for an empty or null state.

The conformance set uses string states only.

### 3.3 Options

Each option has a key, a name, an index (0-based, request order) and a
description:

- choice: name = key; description = the value, or none for `null` or `""`;
- score: name = index as text; description = the level;
- noul: name = key; description = the matching criterion, or none.

A description that is not a string is rendered as JSON as in 3.2. A profile
file declares, per question type, how an option is written:

```json
"options": {
  "choice": {"described": "{name}: {description}", "bare": "{name}"},
  "score":  {"described": "level {index}: {description}"},
  "noul":   {"false": {"described": "false: {description}", "bare": "false: no, the statement does not hold"},
             "true":  {"described": "true: {description}",  "bare": "true: yes, the statement holds"}}
}
```

`described` is used when the option has a description, `bare` otherwise. If
the needed form is not declared, the engine MUST refuse the request.

### 3.4 Instructions

A profile file MAY declare `instructions.fallback`, a text per question type
used when the instructions are empty, and `instructions.required: true`, which
makes an engine refuse a question whose instructions are still empty.

### 3.5 Placeholders

A template string contains placeholders `{name}` with a lowercase name. Filling
replaces every placeholder whose name is given, in a single left-to-right pass:
text inserted from the request is never expanded again, and an unknown
placeholder is left as it is. The placeholders in use are `{type}`,
`{instructions}`, `{name}`, `{description}`, `{index}`, `{state}`, `{label}`,
`{text}` and `{level}`, and in 0.2 `{legend}`, `{system}` and `{user}` (6.7).

### 3.6 Answers

For each question the engine computes a probability for every option, in the
option order (sections 5.5 and 6.5). It answers in the System One format:

| Type | Answer |
| --- | --- |
| choice | `{"type": "choice", "choice": <key of the highest probability>, "probabilities": {key: p}, "confidence": c}` |
| score | `{"type": "score", "score": Σ i·p_i, "legend": {"0": level text, ...}, "probabilities": {"0": p, ...}, "confidence": c}` |
| noul | `{"type": "noul", "noul": p_true}` (and `"confidence"` if declared) |

The argmax takes the first index among equal maxima. Probabilities SHOULD be
rounded to 4 decimals in responses; conformance compares unrounded values. The
response also carries `"usage": {"input_tokens": n, "output_tokens": 0}`.

`confidence` follows the rule the manifest declares per type in
`answers.confidence`, so an engine reports what the model's own code reports:

| Rule | Value for p over n options |
| --- | --- |
| `max-probability` | max p |
| `entropy` | 1 − H(p)/ln n, clipped to [0, 1] (Laya) |
| `typesafe` | (n·max p − 1)/(n − 1), clipped (Decider, choice) |
| `typesafe-ordinal` | 1 − Σ p_i·abs(i − k) / mean_i abs(i − (n−1)/2), k = argmax, clipped (Decider, score) |
| `none` | no confidence field |

## 4. The package

### 4.1 Layout

A package is a directory with `odxp.json` at its root:

```
odxp.json          manifest (section 4.2)
model.onnx          encoder-markers graph ...
model.safetensors   ... and its external data, or
model.gguf          causal-letters weights
tokenizer.json      Hugging Face tokenizers format
template.json       encoder-markers input construction (section 5)
prompt.json         causal-letters prompt (section 6)
calibration.json    temperatures (section 7)
conformance.jsonl   the model's own answers (section 9)
```

File names other than `odxp.json` are free; the manifest names them.

### 4.2 odxp.json

| Field | Required | Meaning |
| --- | --- | --- |
| `standard` | yes | `"odxp/0.1"` or `"odxp/0.2"` (4.5) |
| `name` | yes | the model's name, e.g. `convai-innovations/laya:typed-decisions` |
| `profile` | yes | `encoder-markers` or `causal-letters` |
| `weights` | yes | file entry with `format` (`onnx` or `gguf`); ONNX adds `opset` and `data` (its external data files) |
| `tokenizer` | yes | file entry: `tokenizer.json` |
| `template` | encoder-markers | file entry |
| `prompt` | causal-letters | file entry |
| `calibration` | yes | file entry |
| `answers.confidence` | no | confidence rule per question type (3.6); default `none` |
| `limits` | yes | `max_tokens`, `max_options`; optional `max_levels`, `max_option_tokens`, `max_questions` |
| `conformance` | for compatibility | file entry plus `cases`, `questions`, `errors`, `tolerance`, `generated_by` |
| `source` | no | provenance: original model, licence, converter, export verification |

A **file entry** is `{"path": "...", "sha256": "<64 hex>"}`. A path is relative
to the package root, uses `/`, and MUST NOT be absolute or contain `..`, `\`
or `:`. An engine MUST NOT read any file the manifest does not name, and
SHOULD verify every hash before loading.

`limits.max_tokens` is the longest input the package answers: for
encoder-markers it MUST equal `budgets.total` of the template; for
causal-letters it is the longest row. An engine MAY impose lower limits of its
own and refuse what exceeds them.

### 4.3 Weights

**ONNX** (encoder-markers). The graph MUST have the signature of section 5.4,
opset 17 or later. Its initializers MAY be stored as ONNX external data in any
file the manifest lists under `weights.data`, including the model's published
`model.safetensors`: a safetensors file is an 8-byte little-endian header
length, a JSON header, then each tensor as one contiguous little-endian byte
range, so each ONNX initializer can point at its tensor's offset and length.
A tensor the checkpoint stores in float16 or bfloat16 is then declared in that
type and cast to float32 in the graph, which is exactly what a float32 runtime
does when it loads it. A conversion that does this copies no weights, and the
package's weights hash is the published one.

**GGUF** (causal-letters). The model's GGUF file, unchanged. A model published
without a GGUF at the precision its own code runs is converted from its
weights at that precision (llama.cpp's `convert_hf_to_gguf.py`, F16 for BF16
weights); a quantised GGUF is a derived package (section 10).

### 4.4 In systemone.yaml

A registry manifest (systemone.yaml, manifest spec 0.2) points to the package:

```yaml
runtime:
  standard: odxp/0.1
  package: odxp/        # the directory holding odxp.json
```

### 4.5 Versions

An engine MUST refuse a package whose `standard` it does not implement. Before
1.0, every minor version may change the format: an engine implementing 0.2
runs `odxp/0.1` and `odxp/0.2` packages, and never a newer one. A package
declares the oldest version that has every feature it uses, so that the
engines already deployed keep running it: a converter writes `odxp/0.1`
unless the package uses a feature marked (0.2). The files of a package declare
the same version as its manifest.

## 5. Profile `encoder-markers`

### 5.1 Tokenisation

The engine loads `tokenizer.json` with a Hugging Face `tokenizers`-compatible
tokenizer and encodes every piece of text **without special tokens**
(`add_special_tokens = false`). Special tokens are placed by id, looked up by
their string in the vocabulary. Text equal to a special token's string that
appears inside a piece is tokenised as that token, as `tokenizers` does.

### 5.2 template.json

| Field | Meaning |
| --- | --- |
| `special_tokens` | role → token string. `marker` and `pad` are required; any other role (`cls`, `sep`, ...) may be used in the layout |
| `question_type_ids` | the integer fed as `question_type` for choice, score and noul |
| `layout` | the sequence, in order: `{"token": role}` items and the parts `{"part": "head"}`, `{"part": "options"}`, `{"part": "state"}`, each part exactly once |
| `head` | the question head text, e.g. `"{type} question: {instructions}"` |
| `options` | `prefix` (text put before every option, e.g. `" "`) and the forms of 3.3 |
| `budgets` | `total`, `head`, `option`, and `reserve` (16), `option_floor` (4), `head_floor` (8) |
| `truncation` | `strict` or `truncate-state` (5.3) |
| `reserved_text` | `replace` (default) or `reject`: what the marker token's text in a request does |
| `instructions` | optional fallback and requirement (3.4) |

### 5.3 Input construction

For one question with options o_1..o_k and the state s (all as text):

1. **Texts.** head = fill(`head`, type, instructions); option_i = the form of
   3.3; state = 3.2. If the marker's string occurs in any of them:
   `reject` refuses the request; `replace` replaces each occurrence by one space.
2. **Tokens.** H = encode(head); R_i = encode(`prefix` + option_i); S = encode(state).
3. **Options.** Under `strict`, refuse if any |R_i| > `option`. Each option is
   O_i = [marker] + the first `option` tokens of R_i.
4. **Head budget.** b = `head` − Σ|O_i|. If b < `reserve`, cut every O_i to
   its first max(`option_floor`, ⌊(`head` − `reserve`)/k⌋) tokens and
   recompute b. Under `strict`, refuse if |H| > b or if any O_i was cut
   (|O_i| ≠ |R_i| + 1). Then H is cut to its first max(`head_floor`, b) tokens.
5. **State budget.** F = the number of tokens in every layout item except the
   state; room = `total` − F. Under `strict`, refuse if room < 1 or |S| > room.
   Under `truncate-state`, S is cut to its first max(0, room) tokens.
6. **Assembly.** Concatenate the layout items in order; the options part is
   O_1..O_k, and marker position i is the index of O_i's first token.
7. **Final cut.** If the sequence is longer than `total` (possible only under
   `truncate-state` when F > `total`), it is cut to `total` tokens; if a
   marker falls outside, the request is refused.

The result is (ids, marker positions, question type id). This reproduces
`laya.common.build_sequence` (with `truncate-state`) and Julia's
`julia/data.py sequence(strict=True)` (with `strict`) token for token; the
tests check both on random requests.

### 5.4 Graph

| Name | Type | Shape |
| --- | --- | --- |
| `input_ids` | int64 | [batch, tokens] |
| `attention_mask` | int64 | [batch, tokens], 1 for real tokens, 0 for padding |
| `marker_positions` | int64 | [batch, options] |
| `marker_mask` | bool | [batch, options], true for real options |
| `question_type` | int64 | [batch] |
| `option_logits` (output) | float32 | [batch, options] |

The symbolic dimensions MUST be named `batch`, `tokens` and `options`, so an
engine can pin them for backends that compile fixed shapes (Core ML, QNN):
such an engine pads to a shape bucket and runs one compiled graph per bucket.
Rows are right-padded with the `pad` token (attention 0); padded option slots
have position 0 and mask false. The graph's value at a masked slot is
unspecified. Padding MUST NOT change the logits of real options beyond
floating-point rounding. An engine MAY run the questions of a request in one
batch or one at a time.

### 5.5 Readout

For a question with k options, z = option_logits[row, :k] as float64,
T = the calibrated temperature (section 7), p = softmax(z / T).

## 6. Profile `causal-letters`

### 6.1 Tokenisation

As 5.1: `tokenizer.json`, no special tokens added, no BOS, no chat template.
The context piece and each question piece are encoded **separately** and their
ids concatenated: encoding them as one string gives different ids at the
boundary (for example a state ending in a newline). A typed prompt (6.7)
encodes each row as one string instead, as instruction-tuned models are asked.

### 6.2 prompt.json

| Field | Meaning |
| --- | --- |
| `context` | `template` containing `{state}` (e.g. `"Context:\n{state}"`); optional `max_tokens`, the context ids are cut to their first `max_tokens` |
| `question` | `head` (with `{instructions}`), `option` (with one `{label}` and a `{text}`), `tail` |
| `labels` | the option labels in order, each a single token of the tokenizer, distinct |
| `layouts` | `[{"max_options": n, "encoding": "joined" \| "split-at-label"}]`: the first layout whose `max_options` holds the option count is used |
| `slot` | `"last"`: the answer is read at the last token of the row |
| `options` | the forms of 3.3 |
| `score` | `{"mode": "list"}` or `{"mode": "isolated", "question", "options", "read", "level_strip"}` (6.4) |
| `instructions` | fallback and requirement (3.4) |
| `state` | optional `index_arrays_from`, and (0.2) `json`, `messages`, `empty` (3.2) |
| `decode` | the llama.cpp settings that change the numbers (6.6) |
| `types` | (0.2) a layout per question type, instead of `question`, `labels`, `layouts` and `score` (6.7) |
| `chat` | (0.2) the model's chat template, rendered, and its system text (6.7) |
| `rotations` | (0.2) how the rows of a rotated question are combined, and the prior divided out (6.8) |

### 6.3 Rows

A question with option texts t_1..t_n and question text q is one row:

    ids = encode(fill(context.template, state))[:context.max_tokens] + question piece

- `joined`: question piece = encode(fill(head, q) + Σ_j fill(option, label_j, t_j) + tail).
- `split-at-label`: the option template is split at `{label}` into `before`
  and `after`; question piece = encode(fill(head, q)) + Σ_j (encode(before) +
  [id(label_j)] + encode(fill(after, t_j))) + encode(tail).

Decider uses `joined` up to 10 options (inside one encoded string, `(A` is a
single token) and `split-at-label` beyond, where each label is its own token.

### 6.4 Score questions

- `list`: the levels are the options (forms of 3.3), one row.
- `isolated`: one row per level. The level text is the level's description
  with `level_strip` (a regular expression) removed once from its start; the
  row's question text is fill(`score.question`, instructions, level), its
  options are `score.options` (e.g. `["no", "yes"]`). For level i,
  fit_i = the probability of option `read` in its row; the answer's
  probabilities are fit_i / Σ fit (Σ floored at 1e-9).

### 6.5 Readout

At the slot, z_j = the logit of the token id(label_j) for j < n, as float64;
p = softmax(z / T), T from section 7 (for isolated levels, the score
temperature for the question's level count). Noul reads p over (false, true).

### 6.6 Decoding

Each row MUST be decoded on its own, in one decode call, from empty memory, with
logits requested at the slot only. Packing rows into one decode changes the
probabilities (the Decider authors measured up to 0.02 in BF16). The numbers
also depend on backend settings, which the package declares in
`prompt.json` as the model's own engine uses them:

| `decode` field | Values | llama.cpp setting |
| --- | --- | --- |
| `flash_attention` | `auto` (default), `enabled`, `disabled` | `flash_attn_type` |
| `kv_cache` | `f16` (default), `f32` | `type_k`, `type_v` |
| `ubatch` | integer (default 2048) | `n_ubatch` |

On this repository's measurements, switching flash attention off on the CPU
alone moves Decider's probabilities by up to 0.014 (VALIDATION.md).

### 6.7 Typed layouts (0.2)

A typed prompt lays out each question type on its own, and is how an
instruction-tuned model is asked for a decision without being trained for it
(AnyJev, Appendix A). It declares `types` instead of `question`, `labels`,
`layouts` and `score`, and MAY declare `chat` and `rotations`:

```json
"chat": {"template": "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n",
         "system": "You are a decision function. ..."},
"context": {"template": "State:\n{state}\n\n"},
"types": {
  "choice": {"layouts": [{"max_options": 26, "labels": ["A", "B", "...", "Z"],
                          "head": "Question: {instructions}\nOptions:\n",
                          "option": "{label}. {text}", "separator": "\n",
                          "tail": "\nAnswer with the letter only."}],
             "labels": "positional", "rotate": "cyclic", "listing": "request"},
  "score":  {"layouts": [...], "rotate": "none"},
  "noul":   {"layouts": [{"max_options": 2, "labels": ["No", "Yes"],
                          "head": "Question: {instructions}{legend}\nAnswer ",
                          "option": "{label}", "separator": " or ", "tail": "."}],
             "labels": "attached", "rotate": "cyclic", "listing": ["true", "false"]}
}
```

`types` holds choice, score and noul. For each, the layout used is the first
whose `max_options` holds the option count; its labels MUST be single tokens
of the tokenizer, distinct, at least `max_options` of them. The option
template has one `{label}` and MAY have a `{text}`; the head MAY use
`{instructions}` and `{legend}`.

- `labels`: `positional` (default), the label names the position, so the
  option shown j-th is labelled `labels[j]`; or `attached`, the label names
  the option, so option i (in request order, for noul false then true) is
  labelled `labels[i]` wherever it is shown.
- `listing`: the order the options are first shown in. `request` (default);
  `canonical`, sorted by option text (by code point), ties in request order;
  or, for noul only, the keys in order.
- `rotate`: `none` (default), one row in the listing; or `cyclic`, one row per
  cyclic shift of the listing (6.8).

With the option texts t_i (3.3), the instructions q (3.4), the state text s
(3.2) and a listing π (π_j is the option shown at position j, π⁰ the first
listing), let λ_j be `labels[j]` (positional) or `labels[π_j]` (attached). Then

    question = fill(head, instructions=q, legend=t_{π⁰_0} + t_{π⁰_1} + ...)
               + separator.join(fill(option, label=λ_j, text=t_{π_j}) for each j)
               + tail
    user     = fill(context.template, state=s) + question
    row      = fill(chat.template, system=chat.system, user=user)    (user without chat)

and the row is encoded as one string (6.1). `{legend}` carries the options'
texts, in the first listing, into a head whose options show only their labels:
a noul's descriptions then read as a line each after the question. The answer
slot is the last token, and z_j is the logit of the token of λ_j:
p_pos = softmax(z / T) is a distribution over positions. A choice whose option
texts are not distinct is refused. `context.max_tokens` does not apply.

`chat.template` MUST contain `{user}` once and MAY contain `{system}` once. A
converter renders the model's own chat template once, with marks in place of
the two messages, and replaces them: it MUST check that other messages render
as the template filled with them (a template that changes its messages cannot
be stored this way).

### 6.8 Rotations (0.2)

A model that reads letters prefers some positions and some labels. A question
with `rotate: "cyclic"` and a first listing π⁰ of n options is asked in the n
cyclic shifts π^s_j = π⁰_{(j+s) mod n}, s = 0, ..., n−1, so every option is
shown at every position once (a noul: its two answers in both orders). Each
shift is a row with its distribution p^s over positions; q^s_i = p^s_j for the
j with π^s_j = i is the same distribution over options. The shifts are
combined per option:

- `logmean` (default): z_i = mean_s log max(q^s_i, 10⁻¹²);
  p_i = exp(z_i − max z) / Σ_k exp(z_k − max z). If a position adds a bias in
  logit space, it cancels exactly (Zheng et al., ICLR 2024).
- `mean`: p_i = mean_s q^s_i, normalised.

A question with `rotate: "none"` is combined the same way from its one row.

`rotations.prior` is `{"kind": "none"}` (default) or `{"kind":
"content-free", "probes": [...], "strength": α}` with 0 ≤ α ≤ 1 (Zhao et al.,
ICML 2021). For each shift the row is read once per probe, the probe taking
the place of the state text (so `""` becomes the `empty` text of 3.2); the
prior r^s is the mean of those distributions, each value floored at 10⁻⁸, then
normalised, and before combining

    p^s ← normalise(max(p^s / max((r^s)^α, 10⁻⁸), 10⁻⁸))

elementwise. A probe's row depends on the question and not on the state, so
an engine MAY cache its distribution.

## 7. Calibration

```json
{"standard": "odxp/0.1",
 "temperature": {"choice": 1.64, "score": 1.25, "noul": 1.98},
 "by_option_count": [{"type": "choice", "min": 3, "max": 5, "temperature": 1.76},
                     {"type": "choice", "min": 11, "max": null, "temperature": 0.5}],
 "source": {...}}
```

The temperature for a question of type t with n options is the first
`by_option_count` entry with type t and min ≤ n ≤ max (max null: no bound),
else `temperature[t]`, else 1. Temperatures MUST be positive. `source` records
where the values came from and is not read. A converter writes the values the
native runtime **applies**: Laya clamps every temperature to [0.5, 5.0], so a
shipped 0.1006 becomes 0.5 in calibration.json, and the raw value is kept in
`source`.

### 7.1 Answering at another calibration

A package's temperatures are the ones its model's own code applies; they need
not suit the requests an operator serves. A runtime MAY answer with another
calibration.json (the same schema) in place of the package's. Temperatures
apply after the model's scores, so another calibration changes probabilities
and confidences and not what the model read; for encoder-markers it never
changes which option leads (with rotations or a content-free prior, 6.8, it
can). A runtime answering at another calibration:

- MUST check the package's conformance file at the package's own calibration,
  which the file was recorded with: the check is of the package;
- MUST name the calibration it answers with wherever it describes the model:
  the file's sha256 in discovery (11.3) and in its reports, where a runtime at
  the package's own says `"calibration": "package"`;
- keeps content-free priors (6.8) apart by temperature, since a prior is read
  at its question's temperature.

The file is not part of the package, whose files are hashed in odxp.json; a
publisher who changes a package's own calibration publishes a new package,
with a conformance file its model's code records at the new temperatures.

Fitting (informative): `opendxp calibrate PACKAGE LABELS.jsonl` reads
labelled requests, each a request and a label per question (an option's key,
a distribution over the options, or an answer object, such as another
model's), and writes a calibration.json with, for each question type, the
temperature of least mean KL(label || answer), searched over [0.01, 1000];
`source` records the labels' sha256 and the package's own temperatures. On the
typed-decisions test split (opendxp-paper, E9), the temperatures fitted to
Julia 1's answers on 1,200 labelled training requests take its mean KL from
the gold from 2.78 to 0.23 and its Brier score from 0.336 to 0.114, with the
same decisions.

## 8. Batching and state sharing

A request's questions are independent: the answer to one MUST NOT depend on the
others. An engine MAY batch rows and MAY reuse a shared prefix (the context
piece) across rows only if the results stay within the conformance tolerance.
On llama.cpp's CPU backend, continuing Decider's rows from a shared prefix
moves its probabilities by up to 0.018 and changed one of 91 decisions, a near
tie (VALIDATION.md), so there it is a serving choice and not a conformant way
to run the package.

The same holds for rotations (6.8). Reading fewer shifts than a question
declares, and stopping once the leading option is far enough ahead (AnyJev's
adaptive shifts), changes the probabilities; a prior estimated from earlier
requests (AnyJev's default batch prior) makes one answer depend on other
requests. Both are serving choices and not conformant ways to run a package.

On a GPU the same holds, with other numbers. On an NVIDIA A40, reading a
request's rows together in one call (each its own sequence, from empty memory)
kept every decision and was 1.38 times as fast for Decider; a prompt cache that
decodes only the tokens a row does not share with the one before served
AnyJev's rotations twice as fast, from 32 % of the tokens, with every decision
kept and probabilities up to 0.011 further from the file than without it
(opendxp-paper/FINDINGS.md F13, F16). An engine that serves with such a
setting checks the package with the same setting (`opendxp check
--batch-rows`), and publishes that report.

### 8.1 Precision on accelerators

Accelerators trade precision for speed by default: ONNX Runtime's CUDA provider
multiplies float32 matrices in TF32, and llama.cpp's CUDA backend adds up the
products of F16 and BF16 weights in 16 bits. On an NVIDIA A40 this moved Julia
1's probabilities by up to 0.0067 and AnyJev's by 0.049; asking for float32
products moved them by 0.00005 and 0.008 (AnyJev from its BF16 weights),
costing 1 to 50 % of the speed. Quantized weights (Q8_0, Q4_K_M) run through
llama.cpp's own integer kernels, which this setting does not reach.

A runtime SHOULD let the engine choose between its backend's default
(`fast`) and float32 products (`exact`), and a check report MUST name the
precision it ran at (`runtime.precision`). Neither is required: a package is
compatible on a backend at the precision whose report passes.

## 9. Conformance

### 9.1 conformance.jsonl

One case per line:

```json
{"id": "r01", "tags": ["lang:en"], "request": {...},
 "expected": {"intent": {"type": "choice", "probabilities": {"refund": 0.9412, "cancel order": 0.0321, ...}}}}
{"id": "r22", "request": {...}, "error": {"type": "ValueError", "message": "..."}}
```

`expected` holds, for every question, the probability of every option in
option order as the native runtime reported them (noul as
`{"false": 1 − p, "true": p}`). `error` records that the native runtime
refused the request.

### 9.2 Generation and coverage

A conformance file MUST be generated by running the **native runtime** (the
model's own code, on a CPU) on the requests, and the manifest's
`conformance.generated_by` MUST say which code, which version and which
libraries. The request set is `src/opendxp/data/requests-0.1.jsonl` (52
requests, 91 questions, 11 languages; unchanged in 0.2); a package MAY use
another set if it meets the minimum:

- at least 40 cases;
- at least 5 questions of each type;
- choice questions with every option count from 2 to 10, and one with more than 10;
- at least 3 languages (tagged `lang:<code>`);
- a state of at least 2,000 characters.

The native runtime SHOULD compute in float32 when it records the file, on
unquantized weights, if the model's own code can. A file recorded by a
quantized runtime carries that runtime's arithmetic on that CPU: Decider's,
recorded by its own llama.cpp code on an Apple M4, is missed by 0.056 by the
same package on an x86 CPU, while AnyJev's, recorded by its own code in
float32, passes on both (opendxp-paper/FINDINGS.md F14).

### 9.3 Comparison

An engine **passes a case** when, for every question:

- the option keys are the expected keys, in order;
- max over options of |p_engine − p_expected| ≤ **0.01**;
- the engine's argmax is the expected argmax, or an option whose expected
  probability is within **0.001** of the expected maximum (a tie at the
  recorded precision);

or when the case expects an error and the engine refuses the request. An
engine **passes a conformance file** when it passes every case. The report
(`schemas/check-report.schema.json`) records the cases, the maximum and mean
|dp|, the argmax agreement, the refusals, the latency and the machine.

## 10. Compatibility levels

| Level | Name | Requirement |
| --- | --- | --- |
| 0 | Listed | A manifest; the model runs only with its own code. |
| 1 | Conformant package | The package validates (schemas, hashes, parsers) and its conformance file meets 9.2. |
| 2 | **OpenDXP compatible** | Level 1, and the reference runtime passes the conformance file on a CPU. |
| 3 | OpenDXP portable | Level 2, and the reference runtime also passes on a named accelerator backend (for example Core ML, CUDA, Metal). |

The badge (badge/BADGE.md) certifies level 2. Level 3 names its backends and,
for each, the precision (8.1) and serving settings (8) its report ran at: for
example "CUDA, exact". A level 2 report names the CPU's architecture, since
llama.cpp's CPU kernels differ between ARM and x86.

A **derived package** (quantised or otherwise changed weights) is a different
package: it carries a conformance file generated with the native runtime on
the original weights, and reaches level 2 only if it passes it.

## 11. HTTP binding

A server holds one or more models and answers requests over HTTP. The
reference server is `opendxp serve`.

### 11.1 Endpoints

| Method and path | Body | Answer |
| --- | --- | --- |
| `POST /v1/systemone` | a request (section 3), with an optional `model` | the model's answer (11.2) |
| `GET /v1/models` | none | the models the server holds (11.3) |
| `GET /healthz` | none | `{"ok": true}` while the server can answer |

`/v1/systemone` is the path and body of the System One request that TypeSafe's
Jev API and the System One Engine answer, so a client written for either works
against any OpenDXP server. A server MUST state the protocol version in an
`OpenDXP-Version` header on every answer (`0.2` for this version). Bodies are
JSON in UTF-8.

### 11.2 Answering

The body is a request (section 3). `model` names one of the server's models by
the `id` discovery lists; a server that holds exactly one model MAY answer a
request without it, and one that holds more MUST refuse it. A server MUST
validate the request against `schemas/request.schema.json` before a model sees
it.

A `200` answer is the response of section 3.6 (`schemas/response.schema.json`)
with `model`, the id of the model that answered. It MAY add `latency_ms`.

### 11.3 Discovery

`GET /v1/models` answers `{"object": "list", "data": [...]}`, one entry per
model (`schemas/models.schema.json`): `id`, `"object": "model"`, the package's
`standard` and `profile`, the `question_types` it answers, its `limits`, and
`conformance`: the server's own check of the package (whether it is
compatible, the cases passed, the largest difference, the device), or null
when the server did not run one; and `calibration`: the sha256 and
temperatures of the calibration.json it answers with in place of the
package's (7.1), or null.

### 11.4 Errors

Every non-2xx answer has the body `{"error": {"type": ..., "message": ...}}`
(`schemas/error.schema.json`):

| Status | `type` | When |
| --- | --- | --- |
| 400 | `invalid_request` | the body is not JSON, breaks the request schema, or names no model where one is needed |
| 401 | `unauthorized` | a token is required and missing or wrong |
| 404 | `not_found`, `model_not_found` | an unknown path; an unknown `model` |
| 411 | `invalid_request` | no `Content-Length` |
| 413 | `request_too_large` | the body is over the server's limit |
| 422 | `request_refused` | a valid request the model cannot answer (a limit, a reserved token: the refusals a conformance file records) |
| 503 | `model_not_ready` | the model is still loading; the answer SHOULD carry `Retry-After` |
| 500 | `internal_error` | anything else |

### 11.5 Limits and access

A server MUST accept requests of at least 1 MiB and MAY refuse larger ones. It
MUST read every request body before answering, so a kept-alive connection
never takes an unread body for the next request. A server SHOULD listen only
on the loopback interface unless it requires a bearer token
(`Authorization: Bearer ...`) or runs behind a proxy that authenticates; the
reference server refuses to do otherwise. Browsers MAY be allowed by named
origin (CORS); a server SHOULD NOT allow every origin by default.

## 12. MCP binding

An AI agent reaches a decision model as a tool of the Model Context Protocol
(MCP). The reference server is `opendxp mcp`, on the stdio transport.

### 12.1 Tools

A server exposes one tool per model. With one model the tool is `decide`; with
several, each is `decide_` followed by the model's id, with every character
other than ASCII letters, digits, `_`, `-` and `.` replaced by `_`, cut to 128
characters. A tool's:

- `inputSchema` is the request schema of section 3
  (`schemas/request.schema.json`), so its arguments are a request;
- `outputSchema`, on MCP revisions that have structured output, is the
  response schema of section 3.6;
- `annotations` state `readOnlyHint: true` and `openWorldHint: false`: a
  decision changes nothing and reaches nothing outside the server;
- description names the model and its profile, and says the answer carries a
  calibrated probability per option.

### 12.2 Calls

A call answers the request its arguments carry. The result has one text
content item holding the answer as JSON and, on revisions with structured
output, `structuredContent` with the same answer. A request that breaks the
schema or that the model refuses (11.4's `request_refused`) is a tool
execution error, `isError: true` with a message the agent can act on, not a
protocol error. An unknown tool is the JSON-RPC error `-32602`.

### 12.3 Protocol revisions

A server SHOULD serve the current MCP revision, in which every request carries
its protocol version and client capabilities in `_meta` and `server/discover`
describes the server, and MAY also serve the `initialize` handshake of earlier
revisions for clients that still use it. The reference server does both: it
serves 2026-07-28 per request, and 2025-11-25, 2025-06-18, 2025-03-26 and
2024-11-05 through `initialize`. On stdio a server MUST write nothing but
protocol messages to stdout.

## 13. Security considerations

- Nothing in a package is executed (section 2); templates are data.
- Paths are confined to the package; hashes are checked before loading.
- The marker token cannot be injected through a request: `replace` removes
  it, `reject` refuses it.
- An engine SHOULD bound request sizes (`limits`) and SHOULD bound the
  resources a package's declared budgets imply before loading it.
- A server is an input boundary: it validates every request against the
  schema before a model sees it (11.2), reads and bounds every body (11.5),
  and listens beyond the loopback interface only behind a token or an
  authenticating proxy.
- An MCP tool changes nothing (12.1); an agent may still be steered by what a
  decision returns, so an application acting on one SHOULD threshold on the
  probability for what is at stake.

## 14. Left open

- One pass for all questions (Laya-style batching of every question in one
  sequence) as a declared profile variant; a package runs one row per
  question (or per level, or per rotation).
- Conversation states beyond text turns (content parts, images), and a
  conformance set with JSON and conversation states.
- A rotation budget (reading a subset of the shifts, with a certified rate of
  disagreement with all of them) as a declared variant with its own
  conformance rule, and priors estimated across requests.
- Hidden-state heads (AnyJev's L2: a closed-form head per question, read
  partway down the model) as a profile.
- Vision and audio inputs: a later profile.
- Accelerator numerics: llama.cpp's Metal backend reproduces Decider's
  decisions but not its probabilities within 0.01 (VALIDATION.md). A future
  version may define tolerance classes per backend.
- Quantised ONNX variants and their tolerance.

## Appendix A. The reference conversions

| Model | Profile | How it maps |
| --- | --- | --- |
| Laya (and Laya Studio fine-tunes) | encoder-markers | graph from `laya.common.build_model`; template from `build_sequence` and `render_options` (`truncate-state`, `replace`, budgets from `rl_agent_config.json`); calibration from its temperatures, clamped, with option-count buckets; confidence `entropy` (choice, score) and `max-probability` (noul) |
| Julia 1 (Supersonic Labs) | encoder-markers | graph from the `JuliaDecisionModel` inference subset (last head layer at the markers only); template from `julia/data.py sequence(strict=True)` with the published policy (8192 / 512); an option with no description is its name; temperature 1; confidence `max-probability` |
| Decider (Mark Marosi) | causal-letters | the official GGUF; prompt from decider-ai 1.6.0's plain layout (context and question pieces, A–J joined, 255 single-token labels split at the label beyond 10, isolated score levels with the "Proposed answer" question, the noul fallback question); temperatures by type from `decider_config.json`; confidence `typesafe` / `typesafe-ordinal` |
| AnyJev L0 (Nokia) on Qwen3-1.7B | causal-letters, typed (0.2) | Qwen3-1.7B's weights as an F16 GGUF, decoded with an f32 cache and flash attention off, the closest llama.cpp comes to AnyJev's float32 transformers (Qwen's Q8_0 GGUF is a derived package that does not pass); the typed layouts of anyjev 0.2.0's `build_prompt` with the labels `resolve_labels` picks (A–Z; digits 1–9, or A–J for ten levels; Yes/No attached to their answers); the chat template as `render_chat` renders it (the default system prompt, no thinking); `render_state`; every cyclic shift of a choice and both orders of a noul, combined by log-mean, no prior; temperature 1; confidence `max-probability`. A choice option reads "name" or "name: description", a score level is its description, a noul's descriptions follow its instructions as "Yes: ..." and "No: ..." lines |

## Appendix B. Schemas

| File | Schema |
| --- | --- |
| odxp.json | `schemas/odxp.schema.json` |
| template.json | `schemas/template.schema.json` |
| prompt.json | `schemas/prompt.schema.json` |
| calibration.json | `schemas/calibration.schema.json` |
| one line of conformance.jsonl | `schemas/conformance-case.schema.json` |
| a request | `schemas/request.schema.json` |
| a response | `schemas/response.schema.json` |
| a check report | `schemas/check-report.schema.json` |
| a discovery answer (`GET /v1/models`) | `schemas/models.schema.json` |
| an error answer of the HTTP binding | `schemas/error.schema.json` |
