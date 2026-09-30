# Changelog

## 0.3.1 (2026-09-30)

- **Fixed: `opendxp serve` answered no faster than ~40 ms a request on Linux.**
  An answer's headers and body left in two writes with Nagle's algorithm on, so
  on a kept-alive connection the body waited for the client's delayed ACK of
  the headers. Answers now leave in one write, with `TCP_NODELAY`. Julia 1 on an
  NVIDIA A40, one client: 27.6 decisions/s (p50 62 ms) before, 161.6 (p50
  7.5 ms) after. macOS was not affected.
- **`opendxp serve` reads the requests that wait together** (`--batch N`,
  default 32; 1 answers one at a time as before). A request that finds its
  model idle is read at once, in its own thread, as `predict` reads it; requests
  that arrive while the model reads queue, and one thread reads them together
  (`predict_many`). On an A40 at 64 clients, Julia 1 answered 262 decisions/s
  (78 one at a time; 300 with fused attention), Laya multilingual 191 (87);
  one client, 178 (p50 6 ms). `--batch-rows` decodes a causal-letters
  package's rows together (AnyJev at one client: p50 119 to 66 ms).
- The server's listen backlog is 128; it was socketserver's 5, past which a
  burst of new clients was reset.
- `opendxp bench`'s HTTP client turns Nagle's algorithm off, as curl, requests
  and browsers do: http.client sends a request's headers and body apart.
- Requests are validated with one compiled validator per schema, not a new one
  per request.
- **`opendxp export --opset 23`** (laya, julia): attention as ONNX's fused
  `Attention` operator instead of a chain of small ones, with the mask expanded
  to the shape onnxruntime's kernels take. On an A40 through onnxruntime 1.30,
  6 to 14% more decisions per second one request at a time (Julia 1: 199 to
  210); on an x86 server CPU 1.6 to 2.0 times; on an Apple M4 CPU, Julia 1 from
  26.1 to 30.9. onnxruntime runs the operator on CUDA from 1.30: the runtime
  refuses such a package on CUDA with an older onnxruntime rather than let
  attention fall back to the CPU. encoder-markers reads the opset from the
  manifest's weights entry.

## 0.3.0 (2026-09-30)

- **`opendxp calibrate`** (SPEC.md 7.1): fits a package's temperatures, one per
  question type, to labelled requests (an option's key, a distribution, or an
  answer object per question) and writes a calibration.json, with accuracy, KL,
  Brier and ECE before and after, on the labels and on `--test`. The model is
  read once; each temperature is the least mean KL(label || answer), from a
  log-spaced grid refined by golden-section search. On the typed-decisions test
  split, 50 held-out requests labelled with one option each brought Julia 1's
  mean confidence from 0.96 to 0.72, its accuracy (ECE 0.236 to 0.044), and the
  benchmark's distributions took its KL from the gold from 2.78 to 0.23, with
  the same decisions. `--hard-labels` fits to each label's leading option, to
  how often the model is right, when the labels are distributions.
- **`--calibration FILE`** on `run`, `serve`, `mcp` and `bench`, and
  `load(..., calibration=...)`: answer with another calibration.json than the
  package's. The package's conformance file is still replayed at its own
  (`conformance.replay` included), discovery names the file by its sha256, and
  reports say `"calibration": "package"` otherwise.
- Runtimes have `readouts(items)` (each request read once, answered at any
  calibration) and `distributions(..., calibration)`.
- Fixed: the cache of content-free priors was keyed without the temperature, so
  a runtime answering at two calibrations reused a prior read at the other.

- **`opendxp bench`**: how fast a package (the reference runtime, in this
  process) or any OpenDXP server (over the HTTP binding, at several concurrency
  levels) answers: requests and decisions per second, latency percentiles,
  errors by status and, with `--usd-per-hour`, the cost per 1,000 decisions.
  Its clients back off on 429 and 503 as real ones do. Standard library only.
- **Rows read together** (causal-letters): `batch_rows` decodes up to that many
  rows (a request's questions and rotations) in one llama.cpp call, each its
  own sequence from empty memory; `batch_cache` shares one cache among them or
  gives each its own (`per-row`). It is a serving choice, not part of a package:
  `opendxp check --batch-rows N` says whether it keeps a package compatible on
  a given machine. `predict_many` answers several requests' rows together.
- A request's rows are planned before any is read (typed layouts included),
  then combined; the cache of content-free priors is never filled from a
  planning pass.
- **Requests read together** (encoder-markers): `predict_many` runs several
  requests' questions through the graph in passes of rows of similar length
  (within 1.25 times the shortest, plus 16 tokens; at most 32 rows). Padding is
  masked, so each answer is the one `predict` gives. On an NVIDIA A40, Julia 1's
  89 conformance rows take 243 ms together against 625 ms one at a time; a pass
  filled up to a token budget instead padded short rows to a long one's length
  and was slower than no batching. Static-shape providers (Core ML) answer one
  at a time as before.
- The ONNX exporter names the graph's dimensions (`batch`, `tokens`, `options`)
  instead of declaring `torch.export.Dim` ranges, which torch 2.8 refused for
  models that treat a dimension of 1 specially.
- **No silent fallbacks.** An onnxruntime provider asked for by name that does
  not load (a CUDA 13 build on CUDA 12, say) is an error instead of a run on the
  CPU, and reports list the providers that loaded. A causal-letters run on the
  CPU stays there: a GPU build of llama.cpp no longer offloads its matrix
  products (`op_offload`).
- The encoder converters refuse transformers older than 5.2: 4.57 computes
  ModernBERT differently, and its packages exported without an error, agreed
  with the model they were traced from, and failed their conformance files.
- **Precision** (SPEC.md 8.1): `precision="exact"` (`--precision exact` on
  check, bench, serve and mcp) asks a GPU for float32 products: ONNX Runtime's
  CUDA provider without TF32, llama.cpp's CUDA and HIP backends accumulating
  F16 and BF16 products in float32. On an NVIDIA A40 every encoder package then
  gives the CPU's numbers (5e-5 instead of up to 0.0067), and AnyJev from its
  BF16 weights passes its conformance file (0.0082 instead of 0.049),
  costing 1 to 50 % of the speed. Reports name the precision they ran at.
- `conformance.replay` runs a package's conformance file through a runtime that
  is already loaded, so an engine can check a package as it serves it.
- `opendxp bench` clients keep their connection open between requests, as SDKs
  do.
- `scripts/prefix_sharing.py` works for every layout: it keeps the last row in
  the cache and decodes only the tokens a row adds.

## 0.2.0

OpenDXP 0.2: a wire protocol, and models asked in rotation. Every 0.1 package
still runs, and converters keep writing `odxp/0.1` unless a package needs 0.2.

- **HTTP binding** (SPEC.md 11): `POST /v1/systemone`, `GET /v1/models` with
  each model's conformance summary, `GET /healthz`, an error schema, bearer
  tokens and CORS by named origin. `opendxp serve` implements it with the
  standard library and refuses to listen beyond the machine without a token.
- **MCP binding** (SPEC.md 12): `opendxp mcp` gives every package to AI agents
  as a tool over stdio, with the request and response schemas as its input and
  output schemas. It speaks the current MCP revision (per-request metadata,
  `server/discover`) and the `initialize` handshake of earlier clients.
- **Typed layouts** (SPEC.md 6.7): a causal-letters prompt can lay out each
  question type on its own (labels by position or attached to their options,
  a listing, a legend for descriptions), inside the model's chat template, each
  row encoded whole. States can be rendered as indented JSON or as a
  conversation, with a text for an empty state (3.2).
- **Rotations** (SPEC.md 6.8): a question asked once per cyclic shift of its
  options, the shifts combined by log-mean or mean, with an optional
  content-free prior divided out first.
- `opendxp export anyjev` and `--runtime anyjev`: AnyJev (Nokia) at L0, any
  instruction-tuned model asked the way anyjev 0.2.0 asks it, checked against
  anyjev's own Decider. Validated on Qwen3-1.7B (VALIDATION.md).
- Packages declare `odxp/0.1` or `odxp/0.2`; this implementation runs both,
  and `opendxp validate` checks that a package's files declare its manifest's
  version.

## 0.1.0

The first release of OpenDXP, the Open Decision Exchange Protocol, and its
reference implementation.

- The standard, `odxp/0.1`: packages (`odxp.json` and the files it names by
  SHA-256), the `encoder-markers` and `causal-letters` profiles, calibration,
  conformance files and compatibility levels, with JSON Schemas for every file.
- Reference runtimes on ONNX Runtime and llama.cpp (CPU and GPU builds).
  `--device auto` uses CUDA when available and the CPU otherwise; Core ML,
  OpenVINO, QNN and DirectML are used when named.
- `opendxp export` for Laya, Julia 1 and Decider; `opendxp conformance
  generate` from the models' own code; `opendxp check`, `validate`, `run` and
  `info`.
- The OpenDXP 0.1 request set: 52 requests, 91 questions, 11 languages.
