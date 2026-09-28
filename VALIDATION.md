# Validation

Laya, Julia 1 and Decider converted to OpenDXP 0.1 packages, and AnyJev
(Nokia's L0, on Qwen3-1.7B) to an OpenDXP 0.2 package, and checked against
their own code. Every number here comes from a report in
[validation/](validation/): for each package its `odxp.json`, its template or
prompt, `calibration.json`, `conformance.jsonl` and the check reports. The
weights are not copied there; they are the published files, and each manifest
names them by SHA-256.

Machine: Apple M4 (10 cores), macOS 27, Python 3.12.13, onnxruntime 1.30.0,
llama-cpp-python 0.3.35 (llama.cpp with Metal), 4 threads for llama.cpp. For
AnyJev: anyjev 0.2.0, torch 2.14.0, transformers 5.17.0, 8 threads.

## Method

1. **Convert** with `opendxp export`. The encoder-markers graphs point at the
   checkpoint's `model.safetensors` as ONNX external data (SPEC.md 4.3), so a
   package adds a 3 to 5 MB graph and copies no weights. Decider's package is
   its official Q8_0 GGUF, unchanged. AnyJev's is Qwen3-1.7B's published BF16
   weights as an F16 GGUF (llama.cpp's `convert_hf_to_gguf.py`), since AnyJev
   runs the model's own weights.
2. **Record what the model's own code answers.** `opendxp conformance generate`
   runs the native runtime on the CPU over the OpenDXP 0.1 request set: 52
   requests, 91 questions, 11 languages (Nepali and Thai among them), 2 to 20
   options, a 6,687-character state, and two requests a model may refuse. Laya
   runs through the `laya` package and AnyJev through the `anyjev` package (its
   Decider at L0, on transformers in float32, every rotation read); Julia 1 and
   Decider through `opendxp.native`, rebuilds of their authors' inference code
   (Decider decoding each row in full, as its own GGUF engine does).
3. **Replay it through the reference runtime** with `opendxp check`: on the CPU
   for level 2 (OpenDXP compatible), and on an accelerator backend for level 3.

The native runtimes report probabilities to 4 decimals, so a difference of
5e-5 is the rounding floor: every CPU check below agrees with the model's own
code at the precision it records.

## Conformance

| Package | Profile | Backend | Cases | Questions | max \|dp\| | mean \|dp\| | Argmax | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| laya-typed-decisions | encoder-markers | ONNX Runtime, CPU | 52/52 | 91 | 5.0e-5 | 3.6e-5 | 91/91 | level 2 |
| laya-main | encoder-markers | ONNX Runtime, CPU | 52/52 | 91 | 5.1e-5 | 3.2e-5 | 91/91 | level 2 |
| laya-multilingual | encoder-markers | ONNX Runtime, CPU | 52/52 | 91 | 5.4e-5 | 3.3e-5 | 91/91 | level 2 |
| julia-1 | encoder-markers | ONNX Runtime, CPU | 52/52 | 89 + 2 refusals | 5.2e-5 | 3.0e-5 | 89/89 | level 2 |
| julia-1 | encoder-markers | ONNX Runtime, Core ML | 52/52 | 89 + 2 refusals | 5.2e-5 | 3.1e-5 | 89/89 | level 3 (Core ML) |
| julia-1-official | encoder-markers | ONNX Runtime, CPU | 52/52 | 89 + 2 refusals | 5.2e-5 | 3.0e-5 | 89/89 | level 2 |
| decider-2b | causal-letters | llama.cpp, CPU | 52/52 | 91 | 5.0e-5 | 3.5e-5 | 91/91 | level 2 |
| decider-2b | causal-letters | llama.cpp, Metal | 43/52 | 91 | 0.036 | 0.0041 | 91/91 | not level 3 |
| anyjev-qwen3-1.7b | causal-letters, typed (0.2) | llama.cpp, CPU | 52/52 | 90 + 1 refusal | 5.4e-3 | 2.8e-4 | 90/90 | level 2 |
| anyjev-qwen3-1.7b:q8_0 | causal-letters, typed (0.2) | llama.cpp, CPU | 35/52 | 90 + 1 refusal | 0.25 | 0.015 | 88/90 | derived, not level 2 |

Julia 1 refuses two requests of the set, and the reference runtime refuses the
same two: one puts Julia's marker token in the state (its template says
`reject`; Laya's says `replace`), one has an option longer than Julia's
48-token limit.

AnyJev records its probabilities unrounded, so its differences are llama.cpp's
arithmetic against transformers' float32, not a rounding floor. It refuses one
request of the set, and the reference runtime refuses it too: a noul without
instructions (r51). An AnyJev question is its text, so the mapping and the
package both require instructions. `anyjev-qwen3-1.7b:q8_0` is the same
package with Qwen's own Q8_0 GGUF (below).

`julia-1-official` is the graph Supersonic Labs publish
(`SupersonicLabs/Julia-1-ONNX/model.onnx`), not an export of ours: its inputs
and output renamed to the OpenDXP signature, its 69 initializers pointed at the
checkpoint's safetensors and 96 transposes rebuilt from it (3.0 MB, opset 18).
A model's own ONNX export can become a package without exporting it again.

## Latency

Per request of the set, as the check reports record them (the requests have
one to four questions; mean input 213 to 273 tokens).

| Package | Backend | Load | Request median | Request p90 | Question median |
| --- | --- | --- | --- | --- | --- |
| laya-typed-decisions | CPU | 1.73 s | 128 ms | 341 ms | 71 ms |
| laya-main | CPU | 1.93 s | 128 ms | 436 ms | 78 ms |
| laya-multilingual | CPU | 0.51 s | 48 ms | 89 ms | 26 ms |
| julia-1 | CPU | 0.16 s | 21 ms | 46 ms | 11 ms |
| julia-1 | Core ML | 0.07 s | 42 ms | 114 ms | 21 ms |
| decider-2b | CPU | 1.69 s | 624 ms | 2.84 s | 440 ms |
| decider-2b | Metal | 0.34 s | 343 ms | 1.42 s | 238 ms |
| anyjev-qwen3-1.7b | CPU | 0.30 s | 3.09 s | 9.24 s | 1.55 s |

AnyJev reads a choice once per rotation, so a question with k options is k
rows (the set has 2 to 20), each about 500 tokens with the chat template, each
decoded from empty memory with the f32 cache the package declares. The rows of
a question share their start (system prompt, state, question), which an engine
may decode once only within the tolerance (SPEC.md 8).

Core ML compiles one fixed shape per bucket the first time it meets it; the
mean (262 ms) includes those compilations, the median does not. On this
machine Julia 1 answers faster on the CPU than through Core ML.

## Diagnostics

### Decider's prompt, token for token

`scripts/decider_token_ids.py` builds every row of the request set twice: with
Decider's own prompt code and its Hugging Face tokenizer through
`transformers`, and with the package's `prompt.json` and `tokenizers` alone.
149 rows, 0 differences, and the same option label tokens
([token-ids.json](validation/decider-2b/token-ids.json)).

### AnyJev's prompt, token for token

`scripts/anyjev_token_ids.py` builds every row AnyJev reads for the request
set twice: with anyjev 0.2.0's own `build_prompt`, `render_chat` and
`resolve_labels` and Qwen3's tokenizer through `transformers`, and with the
package's `prompt.json` and `tokenizers` alone. 286 rows (every rotation of
the 90 questions AnyJev takes), 0 differences, and the same label tokens in
every rotation ([token-ids.json](validation/anyjev-qwen3-1.7b/token-ids.json)).
The unit tests go further on a fake model: AnyJev's own Decider and the
reference runtime agree to 1e-12 on random requests and states (text, JSON,
conversations, empty), with and without a content-free prior, combined by
log-mean and by mean, in the request's and the canonical order.

### What moves AnyJev's probabilities

AnyJev's own engine runs the model in float32. The F16 GGUF holds the
published BF16 weights exactly, so what remains is llama.cpp's arithmetic
([check-cpu.json](validation/anyjev-qwen3-1.7b/check-cpu.json),
[default decode](validation/anyjev-qwen3-1.7b/check-cpu-default-decode.json),
[Q8_0](validation/anyjev-qwen3-1.7b/check-cpu-q8_0.json)):

| Weights | Decode | Cases | Argmax | max \|dp\| | mean \|dp\| |
| --- | --- | --- | --- | --- | --- |
| F16 | f32 cache, flash attention off (declared) | 52/52 | 90/90 | 0.0054 | 0.00028 |
| F16 | f16 cache, flash attention auto (llama.cpp's defaults) | 51/52 | 90/90 | 0.012 | 0.00054 |
| Qwen's Q8_0 | declared | 35/52 | 88/90 | 0.25 | 0.015 |

With llama.cpp's defaults one probability (r26, question `cancel`) moves by
0.0116, just past the tolerance; with the cache in f32 and attention unfused
every case passes, so the package declares those settings (SPEC.md 6.6).
Quantising the weights is another matter: Qwen's Q8_0 file changes two
decisions of 90 (r08 `action_needed`, r40 `shipped`) and moves 17 cases past
0.01. It is a derived package (SPEC.md 10), a smaller file for serving that
does not reach level 2.

### Julia 1 on its authors' parity cases

Supersonic Labs publish 100 parity cases with Julia 1. Both graphs, ours and
theirs, give the same argmax on all 100, with logits within 1.2e-4 and
probabilities within 2.2e-5 ([julia-1](validation/julia-1/parity-cpu.json),
[julia-1-official](validation/julia-1-official/parity-cpu.json)).

### What moves a causal-letters model's probabilities

`scripts/llama_numerics.py` decodes 20 rows from five cases on the CPU with the
package's declared settings (flash attention `auto`, f16 cache), then with
other settings and on the GPU
([llama-numerics.json](validation/decider-2b/llama-numerics.json)):

| Against the CPU with declared settings | max \|dp\| | mean of each row's max \|dp\| |
| --- | --- | --- |
| CPU, flash attention off | 0.014 | 0.0027 |
| Metal, declared settings | 0.036 | 0.0093 |
| Metal, flash attention off | 0.037 | 0.0093 |
| Metal, flash attention off, f32 cache | 0.037 | 0.0093 |

The Metal difference does not come from flash attention or the cache's
precision; it is in llama.cpp's Metal kernels. It changes no decision (91/91
in the full check). This is why `prompt.json` declares its decode settings
(SPEC.md 6.6), and why tolerance classes per backend are left open (SPEC.md 14).

### Sharing the prompt's prefix

Every row of a Decider request starts with the same context piece. An engine
can decode that prefix once and continue each row from a copy of it (in
llama.cpp, a sequence copy). `scripts/prefix_sharing.py` runs the reference
runtime both ways over Decider's conformance file, all 52 cases
([CPU](validation/decider-2b/prefix-sharing-cpu.json),
[Metal](validation/decider-2b/prefix-sharing-metal.json)):

| Backend | Decoding | Cases within 0.01 | Argmax | max \|dp\| | mean \|dp\| | Time for the set |
| --- | --- | --- | --- | --- | --- | --- |
| CPU | each row in full | 52/52 | 91/91 | 0 | 0 | 122.9 s |
| CPU | shared prefix | 49/52 | 90/91 | 0.018 | 0.0020 | 67.0 s |
| Metal | each row in full | 43/52 | 91/91 | 0.036 | 0.0041 | 48.9 s |
| Metal | shared prefix | 43/52 | 91/91 | 0.036 | 0.0041 | 30.9 s |

The first row is the reference runtime as SPEC.md 6.6 requires: it reproduces
the file exactly at its 4-decimal precision. On the CPU, sharing is 1.8x
faster, moves probabilities past 0.01 on 3 cases, and changes one decision. In
case r29 (question `topic`), a near tie that the model's own code gives to
"billing" (0.489, against 0.457 for "outage") goes to "outage" (0.475, against
0.470). On Metal it adds nothing measurable to the backend's own difference and
is 1.6x faster. So SPEC.md 8 allows sharing only within the tolerance, and the
reference runtime decodes every row in full. Sharing is a serving choice, outside conformance.

## Reproduce

The checkpoints are the makers' own, from Hugging Face: convaiinnovations/laya
(the root and its `multilingual/` and `typed-decisions/` folders),
SupersonicLabs/Julia-1, SupersonicLabs/Julia-1-ONNX (`model.onnx`,
`parity-cases.json`), Mapika/decider-2b-GGUF (`decider-2b-v11-Q8_0.gguf`
with its tokenizer and `decider_config.json`), and for AnyJev Qwen/Qwen3-1.7B
(revision 70d244c) and Qwen/Qwen3-1.7B-GGUF (`Qwen3-1.7B-Q8_0.gguf`), with
anyjev 0.2.0 from PyPI.

```bash
opendxp export laya ckpt/laya/typed-decisions packages/laya-typed-decisions --name convai-innovations/laya:typed-decisions
opendxp conformance generate packages/laya-typed-decisions --native ckpt/laya/typed-decisions --runtime laya
opendxp check packages/laya-typed-decisions --report validation/laya-typed-decisions/check-cpu.json
opendxp export julia ckpt/julia-1 packages/julia-1-official --official-onnx ckpt/julia-1-onnx/model.onnx
opendxp check packages/julia-1 --device coreml --report validation/julia-1/check-coreml.json

python scripts/decider_token_ids.py packages/decider-2b ckpt/decider-2b
python scripts/julia_parity.py packages/julia-1 ckpt/julia-1-onnx/parity-cases.json
python scripts/llama_numerics.py packages/decider-2b
python scripts/prefix_sharing.py packages/decider-2b --device cpu --report validation/decider-2b/prefix-sharing-cpu.json

python llama.cpp/convert_hf_to_gguf.py ckpt/qwen3-1.7b --outtype f16 --outfile ckpt/Qwen3-1.7B-F16.gguf
opendxp export anyjev ckpt/qwen3-1.7b packages/anyjev-qwen3-1.7b --gguf ckpt/Qwen3-1.7B-F16.gguf --base Qwen/Qwen3-1.7B
opendxp conformance generate packages/anyjev-qwen3-1.7b --native ckpt/qwen3-1.7b --runtime anyjev --threads 8
opendxp check packages/anyjev-qwen3-1.7b --threads 8 --report validation/anyjev-qwen3-1.7b/check-cpu.json
python scripts/anyjev_token_ids.py packages/anyjev-qwen3-1.7b ckpt/qwen3-1.7b
```

## Not measured yet

- CUDA, ROCm (MIGraphX), OpenVINO, QNN and DirectML: no such hardware here.
- An x86 server CPU (AVX-512). The first System One Engine deployment on one
  is the next measurement.
- Laya on Core ML. ONNX Runtime's Core ML provider compiles a Core ML model
  for each input shape, and the reference runtime keeps shapes to a set of
  buckets. For Laya (421M parameters) the compiled models had filled 2.3 GB of
  temporary space when the run was stopped for lack of disk. Fewer, larger
  buckets, or one compilation with flexible shapes, is the next thing to try.
- Quantised ONNX variants of the encoders.
