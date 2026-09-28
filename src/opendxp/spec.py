"""The constants of the standard: its version, profiles, question types and tolerances.

Everything here is normative: SPEC.md states the same values, and a change to
any of them is a change to the standard.
"""

from __future__ import annotations

STANDARD = "odxp/0.1"
MAJOR, MINOR = 0, 1

PROFILES = ("encoder-markers", "causal-letters")
QUESTION_TYPES = ("choice", "score", "noul")

# Conformance: the largest absolute difference allowed between a probability the
# reference runtime computes and the one the model's own code recorded, and the
# margin under which two expected probabilities count as tied for the argmax test.
TOLERANCE = 0.01
TIE_MARGIN = 0.001

# The minimum a conformance file must cover before a package can be called compatible.
MIN_CASES = 40

# Answers round probabilities to this many decimals, as the System One format does.
DECIMALS = 4

# Confidence rules a package may declare for its answers (SPEC.md, "Answers").
CONFIDENCE_RULES = ("max-probability", "entropy", "typesafe", "typesafe-ordinal", "none")


def parse_standard(value: object) -> tuple[int, int]:
    """(major, minor) of an "odxp/X.Y" string, or ValueError."""
    if not isinstance(value, str) or not value.startswith("odxp/"):
        raise ValueError(f"not an OpenDXP standard version: {value!r}")
    try:
        major, minor = (int(x) for x in value.removeprefix("odxp/").split("."))
    except ValueError:
        raise ValueError(f"not an OpenDXP standard version: {value!r}") from None
    return major, minor


def supported(value: object) -> bool:
    """Whether this implementation can run a package declaring `value`.

    Before 1.0 every minor version may change the format, so an engine runs
    exactly the minor versions it implements, and never a newer one.
    """
    try:
        major, minor = parse_standard(value)
    except ValueError:
        return False
    return (major, minor) == (MAJOR, MINOR)
