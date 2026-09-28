# Changelog

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
