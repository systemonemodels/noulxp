# OpenDXP

**The Open Decision Exchange Protocol: a portable standard for calibrated single-pass decision models.**

A System One model reads a state and answers typed questions (choice, score,
noul) with a calibrated probability for every option, in one pass. Every such
model ships its own inference code: Laya needs the `laya` package, Julia 1 its
own package, Decider its own server. To run twenty models you install twenty
runtimes, and to compare them you write twenty harnesses.

OpenDXP makes the model a package that one engine runs without code written for
it. A package declares which of two **profiles** it follows, carries its
weights in a portable format (ONNX or GGUF), describes its input construction
and calibration as data, and ships a **conformance file**: what the model's
own code answers on a fixed set of requests. An engine that reproduces that
file within 0.01 in probability, with the same decisions, runs the model
faithfully.

Requests travel the same way everywhere: over HTTP between an application and
any server (`opendxp serve`, or an engine such as the System One Engine), and
over the Model Context Protocol between an AI agent and a tool (`opendxp mcp`).
It is to decision models what MCP is to agent tools: one way to ask any model,
on any machine.

- [SPEC.md](https://github.com/systemonemodels/opendxp/blob/main/SPEC.md): the standard, version 0.2
- [schemas/](https://github.com/systemonemodels/opendxp/tree/main/schemas): JSON Schemas for every file in a package
- [VALIDATION.md](https://github.com/systemonemodels/opendxp/blob/main/VALIDATION.md): Laya, Julia 1, Decider and AnyJev converted and checked
- [badge/BADGE.md](https://github.com/systemonemodels/opendxp/blob/main/badge/BADGE.md): the "OpenDXP compatible" badge and its criteria
- [PAPER.md](https://github.com/systemonemodels/opendxp/blob/main/PAPER.md): outline of the paper, and which experiments are done

OpenDXP is open: the specification, the schemas and this reference
implementation are Apache-2.0, and anyone may build an engine for it. The
[System One Engine](https://systemonemodels.tech/docs/engine), which serves
the live playgrounds on [System One Models](https://systemonemodels.tech),
runs OpenDXP packages and shows which models are OpenDXP compatible.

## The two profiles

| Profile | Models | Weights | The engine reads |
| --- | --- | --- | --- |
| `encoder-markers` | Laya, Julia 1, Von, open-jev, GLiNER2.5-Decide | ONNX | `template.json`: how to build the token ids and one marker position per option |
| `causal-letters` | Decider, AnyJev, Kev, Nimble, JevK5, lev | GGUF | `prompt.json`: the prompt pieces, the option labels and where the answer letter is read |

ONNX runs through ONNX Runtime (CPU, CUDA, Core ML, OpenVINO, QNN, DirectML);
GGUF through llama.cpp (CPU, Metal, CUDA, HIP, SYCL, Vulkan). The engine picks
the device, not the model's author.

New in 0.2, a causal-letters prompt can lay out each question type on its own,
inside the model's chat template, and ask a question once per cyclic shift of
its options, combining the shifts per option so that no option wins by its
position. That is how [AnyJev](https://github.com/nokia-applied-research/AnyJev)
(Nokia) turns an instruction-tuned language model into a decision model
without training it: `opendxp export anyjev` packages such a model, and
AnyJev's own code writes its conformance file.

## What is here

The `opendxp` Python package (Python 3.11+) is the reference implementation:

- **Reference runtimes** for both profiles. They read only the package and
  contain nothing specific to any model. The encoder runtime runs on CUDA when
  onnxruntime has it and on the CPU otherwise; Core ML, OpenVINO, QNN and
  DirectML are used when named (`--device coreml`), and it compiles shape
  buckets for providers that need fixed shapes. The causal runtime drives llama.cpp's low-level API, one
  row per decode, with every layer on the GPU when the build has one.
- **Converters** for Laya, Julia 1, Decider and AnyJev (`opendxp export`).
- **The models' own inference** for writing conformance files
  (`opendxp.native`): Laya through the `laya` package, AnyJev through the
  `anyjev` package; Julia 1 and Decider through faithful rebuilds of their
  authors' code, checked on the authors' published cases.
- **Servers** for the two bindings: `opendxp serve` answers requests over HTTP
  on any machine, and `opendxp mcp` gives every package to AI agents as an MCP
  tool.
- **Conformance tools**: `opendxp conformance generate` runs the model's own
  code on the OpenDXP 0.1 request set (52 requests, 91 questions, 11 languages
  including Nepali and Thai); `opendxp check` replays it through the reference
  runtime and writes a JSON report.

## Install

```bash
pip install "opendxp[onnx]"          # encoder-markers runtime (ONNX Runtime)
pip install "opendxp[gguf]"          # causal-letters runtime (llama-cpp-python)
pip install "opendxp[export,laya]"   # converters (torch, transformers, onnx, onnxscript, laya)
pip install "opendxp[gguf,anyjev]"   # AnyJev: its converter and its own Decider (anyjev, torch, transformers)
```

From a clone, for development:

```bash
pip install -e ".[dev]"             # tests and linting
```

## Use a package

```python
import opendxp

model = opendxp.load("packages/laya-typed-decisions")      # device="auto": the best available
out = model.predict(
    "Customer: I was charged twice and still have no refund.",
    {
        "intent": {"type": "choice", "instructions": "What does the customer want?",
                   "criteria": {"refund": None, "cancel order": None, "other": None}},
        "urgency": {"type": "score", "instructions": "How urgent is this?",
                    "criteria": ["low", "medium", "high"]},
        "angry": {"type": "noul", "instructions": "The customer is angry."},
    },
)
# {"answers": {"intent":  {"type": "choice", "choice": ..., "probabilities": {...}, "confidence": ...},
#              "urgency": {"type": "score", "score": ..., "legend": {...}, "probabilities": {...}, ...},
#              "angry":   {"type": "noul", "noul": ..., ...}},
#  "usage": {"input_tokens": ..., "output_tokens": 0}}
```

From the shell: `opendxp run PACKAGE --request request.json [--device cpu|coreml|cuda|gpu]`.
`opendxp info` lists this machine's backends.

## Serve it

```bash
opendxp serve packages/laya-typed-decisions packages/julia-1
```

Any machine now answers the System One request over HTTP (SPEC.md 11), the
same request and path TypeSafe's Jev API and the System One Engine answer, so
their clients work unchanged:

```bash
curl -s localhost:8790/v1/models          # what this server holds
curl -s localhost:8790/v1/systemone -H 'content-type: application/json' -d '{
  "model": "supersonic-labs/julia-1",
  "state": "Customer: I was charged twice and still have no refund.",
  "questions": {"angry": {"type": "noul", "instructions": "The customer is angry."}}
}'
```

It listens on 127.0.0.1:8790. To listen anywhere else it needs a token
(`--token`, or `OPENDXP_TOKEN`), which clients send as
`Authorization: Bearer ...`. `--check` runs each package's conformance file at
start and reports the result in `/v1/models`; `--cors ORIGIN` lets a browser
page call it.

## Measure it

```bash
opendxp bench path/to/package --device cuda --rounds 3 --usd-per-hour 0.49
opendxp bench http://127.0.0.1:8790 --model nokia/anyjev-qwen3-1.7b --concurrency 1,4,16,64
```

`bench` reports requests and decisions per second (a decision is one question
answered), latency percentiles and, given the machine's price per hour, the
cost per 1,000 decisions, for a package in this process or for any server that
speaks the HTTP binding (`opendxp serve`, an engine, a hosted API). Publish it
with `opendxp check` on the same machine: speed means nothing without the
answers being the model's own.

A causal-letters package can read a request's rows together:
`--batch-rows 16` (and `--batch-cache per-row`), for `check`, `bench` and the
Python runtime. Whether that keeps a package compatible on a given machine is
what `opendxp check --batch-rows 16` answers.

GPUs trade precision for speed by default (TF32 in ONNX Runtime, 16-bit
accumulation in llama.cpp's CUDA backend). `--precision exact` asks for
float32 products on `check`, `bench`, `serve` and `mcp`: on an NVIDIA A40 it
brought every encoder package to the CPU's numbers and made AnyJev's BF16
package pass on CUDA, costing 1 to 50 % of the speed (SPEC.md 8.1). Check at the
precision you serve with.

## Calibrate it to your data

A package's temperatures are data (SPEC.md 7), so how sure a model says it is
can be fitted to the requests you serve without touching its weights:

```bash
opendxp calibrate packages/julia-1 labelled.jsonl --test held-out.jsonl --out julia-cal.json
opendxp serve packages/julia-1 --calibration julia-cal.json
```

`labelled.jsonl` is one request per line with a label per question: an
option's key, a distribution over the options, or an answer object (another
model's, say):

```json
{"state": "I was charged twice.", "questions": {"intent": {"type": "choice", "criteria": ["refund", "cancel"]}, "angry": {"type": "noul"}}, "labels": {"intent": "refund", "angry": false}}
```

It fits one temperature per question type (the least mean KL from the labels,
the log loss for one-hot labels), prints confidence, accuracy, KL, Brier and ECE
before and after, and writes a calibration.json whose `source` records the
labels' hash. Label with what was right (one option per question) to make the
model's confidence match how often it is right; label with distributions, such
as a bigger model's answers, to match their spread. Use requests the model was
not trained on.
The package is untouched: `opendxp check` still checks it at its own
temperatures, and a server answering at a fitted file names it (its sha256) in
`/v1/models`. On the typed-decisions test split, Julia 1 answers with a mean
confidence of 0.96 and is right 72 % of the time; fitted to 50 held-out
requests labelled with one option each, its confidence came to 0.72 (ECE 0.236
to 0.044), with the same decisions. `run`, `serve`, `mcp` and `bench` take
`--calibration`; in Python, `opendxp.load(path, calibration=...)`.

## Give it to an AI agent (MCP)

`opendxp mcp` serves packages as tools of the Model Context Protocol, on
stdio (SPEC.md 12): an agent calls `decide` with a request and gets the answer,
with a calibrated probability per option, as structured output. In an MCP
client's configuration (Claude Desktop, Claude Code, Cursor and others):

```json
{
  "mcpServers": {
    "julia-1": {
      "command": "opendxp",
      "args": ["mcp", "/path/to/packages/julia-1"]
    }
  }
}
```

It speaks both eras of MCP: the current revision (per-request metadata,
`server/discover`) and the `initialize` handshake that earlier clients use.

## Convert a model and check it

```bash
# 1. Convert: the package's graph references the checkpoint's weights, so it adds a few MB.
opendxp export laya    ckpt/laya-typed-decisions   packages/laya-typed-decisions
opendxp export julia   ckpt/julia-1                packages/julia-1
opendxp export decider ckpt/decider-2b-gguf        packages/decider-2b
opendxp export anyjev  ckpt/qwen3-1.7b             packages/anyjev-qwen3-1.7b \
    --gguf ckpt/Qwen3-1.7B-F16.gguf --base Qwen/Qwen3-1.7B   # llama.cpp's convert_hf_to_gguf.py --outtype f16

# 2. Record what the model's own code answers (on the CPU).
opendxp conformance generate packages/julia-1 --native ckpt/julia-1 --runtime julia

# 3. Replay it through the reference runtime.
opendxp check packages/julia-1                    # the reference: CPU
opendxp check packages/julia-1 --device coreml    # a backend: Core ML
opendxp validate packages/julia-1                 # schemas, parsers, coverage, hashes
```

`--runtime laya` uses the `laya` package (the `laya` extra) and
`--runtime anyjev` the `anyjev` package (the `anyjev` extra); `julia` and
`decider` use `opendxp.native` (with the `export` extra, and llama-cpp-python
for Decider).

## Make your model OpenDXP compatible

1. **Pick the profile.** If your model scores a marker token per option with a
   bidirectional encoder, it is `encoder-markers`. If it is a language model
   that reads the probability of option letters after a prompt, it is
   `causal-letters`; an instruction-tuned model asked in its chat template,
   in rotation, uses the typed layouts of SPEC.md 6.7 and 6.8.
2. **Export the weights.** encoder-markers: ONNX with the signature in SPEC.md
   5.4 (`input_ids`, `attention_mask`, `marker_positions`, `marker_mask`,
   `question_type` → `option_logits`), dynamic dimensions named `batch`,
   `tokens`, `options`. `opendxp.export.onnx_graph.export_graph` does this for
   any torch module with that forward, and points the graph at your
   `model.safetensors` instead of copying it. causal-letters: your GGUF file.
3. **Describe the input as data.** Write `template.json` or `prompt.json`
   (SPEC.md sections 5 and 6): the head and option texts, the special tokens,
   the budgets and truncation rule, or the prompt pieces, labels and score
   mode. If your input cannot be described this way, open an issue: that is
   what the next version of the standard needs to know.
4. **Declare calibration**: the temperatures your code applies, by type and
   option count (section 7).
5. **Generate conformance with your own code.** Add a native adapter in
   `src/opendxp/native/` that loads your model with your package and returns
   System One answers, then run `opendxp conformance generate`. The file records
   which code produced it.
6. **Check.** `opendxp check` must pass on the CPU (level 2, "OpenDXP compatible";
   SPEC.md section 10). Put the report next to the package and the badge in
   your model card.

## Tests

```bash
pytest
```

The tests port the models' own input construction verbatim (Laya's
`build_sequence`, Julia's `sequence()`, Decider's prompt builder) and require
the declarative templates to give the same token ids, marker positions and
refusals on hundreds of random requests. For AnyJev they run its own code
(its core is numpy alone): its prompts, chat rendering and Decider on a fake
model, which the typed layouts and rotations must match row for row and to
1e-12 in probability. A toy ONNX graph with the standard signature exercises
generate and check end to end. Nothing is downloaded.

## Licence

Apache-2.0. Converted packages carry the models' own weights and remain under
the models' licences (see NOTICE).
