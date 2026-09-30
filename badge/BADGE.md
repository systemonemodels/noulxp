# The NoulXP badge

![NoulXP compatible](noulxp-compatible.svg)

The badge says one thing: **the reference NoulXP runtime runs this package and
reproduces the model's own answers.** It is SPEC.md level 2.

## Criteria

A package earns "NoulXP compatible" when all of these hold:

1. **It validates.** `noulxp validate PACKAGE` reports nothing: the manifest,
   `template.json` or `prompt.json` and `calibration.json` match their schemas
   and parse; every file matches its SHA-256; nothing in the package is
   executable.
2. **Its conformance file was made by the model's own code.** The manifest's
   `conformance.generated_by` names the native runtime, its version and the
   libraries it ran with, on a CPU.
3. **The conformance file covers enough** (SPEC.md 9.2): at least 40 cases; 5 or
   more questions of each type; choice questions with 2 to 10 options and one
   with more than 10; 3 or more languages; a state of 2,000 characters or more.
   The NoulXP 0.1 request set (52 requests, 91 questions, 11 languages) meets it.
4. **The reference runtime passes it on a CPU** (SPEC.md 9.3): every question
   within 0.01 of every expected probability, the same argmax (ties within
   0.001 accept either option), and every expected refusal refused.
   `noulxp check PACKAGE --device cpu` prints `PASS` and its report has
   `"compatible": true`.

The badge links to the check report, which names the package's hashes, so a
changed package needs a new check.

![NoulXP portable](noulxp-portable.svg)

**"NoulXP portable"** (level 3) adds: the reference runtime also passes the same
conformance file on a named accelerator backend (Core ML, CUDA, Metal,
OpenVINO, ...). A portable badge always lists its backends next to it.

## Using it

```markdown
[![NoulXP compatible](https://.../noulxp-compatible.svg)](link-to/check-cpu.json)
```

Keep `check-cpu.json` (and any backend reports) next to the package.

## What the badge does not say

It does not rate accuracy, calibration quality, safety or speed. It says only
that any NoulXP engine can run the model and get the answers its authors' code
gives. Comparing models is what that makes possible; the badge is not the
comparison.
