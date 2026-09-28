# Paper outline

Working title: **OpenDXP: a portable runtime standard for calibrated
single-pass decision models**

This file tracks the paper: its argument, its sections, and which experiments
are done. Numbers come from [VALIDATION.md](VALIDATION.md) and are measured,
never estimated.

## The argument

1. A new kind of model answers typed questions about a state (choose one of
   these options, score on this scale, is this true) with a calibrated
   probability for every option, in one forward pass. Laya, Julia 1 and
   Decider are three of them, from three teams, with three architectures.
2. Each ships its own inference code. Running, deploying or comparing them
   means installing, trusting and wrapping each one. Formats such as ONNX and
   GGUF carry the weights but not the rest: how the input is built from the
   request, where the answer is read, how it is calibrated.
3. OpenDXP puts the rest in the package as data (a template or a prompt, and
   temperatures), so one engine runs every model with no code written for it
   and nothing from the package executed.
4. Faithfulness is testable: the package carries a conformance file of what
   the model's own code answers, and an engine passes when it reproduces it
   within 0.01 with the same decisions.
5. Two profiles cover the models we know of: bidirectional encoders that
   score a marker token per option, and causal language models read at an
   answer letter.

## Sections

1. **Introduction.** System One models; the cost of one runtime per model;
   the contribution: the standard, the conformance method, a reference
   implementation and three conversions.
2. **Background.** Calibrated classification and temperature scaling (Guo et
   al., 2017); marker-based encoders (Laya, Julia 1); letter-probability
   readout from language models (Decider); ONNX and GGUF as weight formats;
   loading model code at run time (`trust_remote_code`) and why a standard
   should not need it.
3. **The standard.** Requests and answers; packages with nothing executable;
   the two profiles; calibration as data; conformance files and the
   comparison rule; compatibility levels.
4. **Reference implementation.** Runtimes for both profiles on ONNX Runtime
   and llama.cpp; converters that copy no weights (ONNX external data pointing
   into the published safetensors); conformance generation from the models'
   own code; the System One Engine and the registry, which run packages and
   show the badge.
5. **Evaluation.** The experiments below.
6. **Discussion.** What the tolerance means; backend numerics; sharing
   computation versus exactness; what 0.1 leaves open (SPEC.md 14).
7. **Conclusion.**

## Experiments

| # | Question | Status |
| --- | --- | --- |
| E1 | Do packages reproduce their models' own answers on a CPU? | **Done.** 6 packages, 3 families, 52/52 cases each, all decisions equal, differences at the 4-decimal rounding floor. |
| E2 | Does a declarative input description give the same tokens as the models' own code? | **Done.** Decider: 149/149 rows identical. Laya and Julia: input construction tested against their own code (tests/). |
| E3 | Can a model's own ONNX export become a package without exporting again? | **Done.** Julia 1's published graph, renamed and pointed at its checkpoint: 52/52, and 100/100 on its authors' parity cases. |
| E4 | Do packages stay faithful on accelerator backends? | **Partly.** Julia on Core ML passes (level 3). Decider on Metal keeps every decision but not the probabilities (max 0.036). CUDA, ROCm, OpenVINO, QNN, DirectML to do. |
| E5 | What moves a causal model's probabilities? | **Done on one machine.** Flash attention off on the CPU: up to 0.014. Metal: 0.036 whatever the attention or cache settings. |
| E6 | What does sharing the prompt prefix cost in faithfulness? | **Done on one machine.** CPU: 1.8x faster over the set, up to 0.018, and one near-tie decision of 91 changed; Metal: 1.6x faster, nothing added. To repeat on an x86 server. |
| E7 | What does a package cost? | **Done on one machine.** 3 to 5 MB of graph per encoder package, no weight copies; load and latency per backend. Server numbers to do. |
| E8 | Is 0.01 the right tolerance? | **To do.** The distribution of differences, and of decision flips, across backends and quantisations. |
| E9 | Can someone else convert a model from the spec alone? | **To do.** A Laya Studio fine-tune and a model from outside the three families, converted by their authors. |

## Before submission

- E4, E6 and E7 on an x86 server CPU and at least one NVIDIA GPU.
- E8 with quantised variants (int8 ONNX, Q4 GGUF) as derived packages.
- E9 with at least one team outside System One Models.
- Credit and permission: the model authors (Convai Innovations, Supersonic
  Labs, Mark Marosi) are named for their models and asked before their
  results are published.
