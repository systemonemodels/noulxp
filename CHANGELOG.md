# Changelog

## 0.1.0

The first release of OpenDXP, the Open Decision Exchange Protocol, and its
reference implementation.

- The standard, `odxp/0.1`: packages (`odxp.json` and the files it names by
  SHA-256), the `encoder-markers` and `causal-letters` profiles, calibration,
  conformance files and compatibility levels, with JSON Schemas for every file.
- Reference runtimes on ONNX Runtime (CPU, CUDA, Core ML, OpenVINO, QNN,
  DirectML) and llama.cpp (CPU and GPU builds).
- `opendxp export` for Laya, Julia 1 and Decider; `opendxp conformance
  generate` from the models' own code; `opendxp check`, `validate`, `run` and
  `info`.
- The OpenDXP 0.1 request set: 52 requests, 91 questions, 11 languages.
