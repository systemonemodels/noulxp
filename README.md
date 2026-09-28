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

- [SPEC.md](https://github.com/systemonemodels/opendxp/blob/main/SPEC.md): the standard, version 0.1
- [schemas/](https://github.com/systemonemodels/opendxp/tree/main/schemas): JSON Schemas for every file in a package
- [VALIDATION.md](https://github.com/systemonemodels/opendxp/blob/main/VALIDATION.md): Laya, Julia 1 and Decider converted and checked
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
| `causal-letters` | Decider, Kev, Nimble, JevK5, lev | GGUF | `prompt.json`: the prompt pieces, the option labels and where the answer letter is read |

ONNX runs through ONNX Runtime (CPU, CUDA, Core ML, OpenVINO, QNN, DirectML);
GGUF through llama.cpp (CPU, Metal, CUDA, HIP, SYCL, Vulkan). The engine picks
the device, not the model's author.

## What is here

The `opendxp` Python package (Python 3.11+) is the reference implementation:

- **Reference runtimes** for both profiles. They read only the package and
  contain nothing specific to any model. The encoder runtime chooses an
  onnxruntime execution provider from what is installed (CUDA, Core ML,
  OpenVINO, QNN, DirectML, CPU) and compiles shape buckets for providers that
  need fixed shapes. The causal runtime drives llama.cpp's low-level API, one
  row per decode, with every layer on the GPU when the build has one.
- **Converters** for Laya, Julia 1 and Decider (`opendxp export`).
- **The models' own inference** for writing conformance files
  (`opendxp.native`): Laya through the `laya` package; Julia 1 and Decider
  through faithful rebuilds of their authors' code, checked on the authors'
  published cases.
- **Conformance tools**: `opendxp conformance generate` runs the model's own
  code on the OpenDXP 0.1 request set (52 requests, 91 questions, 11 languages
  including Nepali and Thai); `opendxp check` replays it through the reference
  runtime and writes a JSON report.

## Install

```bash
pip install "opendxp[onnx]"          # encoder-markers runtime (ONNX Runtime)
pip install "opendxp[gguf]"          # causal-letters runtime (llama-cpp-python)
pip install "opendxp[export,laya]"   # converters (torch, transformers, onnx, onnxscript, laya)
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

## Convert a model and check it

```bash
# 1. Convert: the package's graph references the checkpoint's weights, so it adds a few MB.
opendxp export laya    ckpt/laya-typed-decisions   packages/laya-typed-decisions
opendxp export julia   ckpt/julia-1                packages/julia-1
opendxp export decider ckpt/decider-2b-gguf        packages/decider-2b

# 2. Record what the model's own code answers (on the CPU).
opendxp conformance generate packages/julia-1 --native ckpt/julia-1 --runtime julia

# 3. Replay it through the reference runtime.
opendxp check packages/julia-1                    # the reference: CPU
opendxp check packages/julia-1 --device coreml    # a backend: Core ML
opendxp validate packages/julia-1                 # schemas, parsers, coverage, hashes
```

`--runtime laya` uses the `laya` package (the `laya` extra);
`julia` and `decider` use `opendxp.native` (with the `export` extra, and
llama-cpp-python for Decider).

## Make your model OpenDXP compatible

1. **Pick the profile.** If your model scores a marker token per option with a
   bidirectional encoder, it is `encoder-markers`. If it is a language model
   that reads the probability of option letters after a prompt, it is
   `causal-letters`.
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
refusals on hundreds of random requests. A toy ONNX graph with the standard
signature exercises generate and check end to end. Nothing is downloaded.

## Licence

Apache-2.0. Converted packages carry the models' own weights and remain under
the models' licences (see NOTICE).
