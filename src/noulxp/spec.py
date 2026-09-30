"""The constants of the standard: its version, profiles, question types and tolerances.

Everything here is normative: SPEC.md states the same values, and a change to
any of them is a change to the standard.
"""

from __future__ import annotations

# The versions this implementation runs, oldest first. A package declares the
# oldest version that has every feature it uses (SPEC.md 4.5), so converters
# write BASE unless a package needs more.
VERSIONS = ("noulxp/0.1", "noulxp/0.2")
BASE, STANDARD = VERSIONS[0], VERSIONS[-1]

# Until 0.3.1 the standard was called OpenDXP and packages declared "odxp/X.Y".
# Those are the same versions under the old name: every runtime still reads
# them, and nothing writes them any more.
PREFIX, LEGACY_PREFIX = "noulxp/", "odxp/"
LEGACY_VERSIONS = tuple(LEGACY_PREFIX + v.removeprefix(PREFIX) for v in VERSIONS)

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

# The HTTP binding (SPEC.md 11): the protocol version a server states in every
# answer, its two paths, the largest request it must accept, and the reference
# server's default port. The path is the one Jev's API and the System One Engine
# already answer on, so their clients work against any NoulXP server.
PROTOCOL_VERSION = "0.2"
HTTP_DECIDE_PATH = "/v1/systemone"
HTTP_MODELS_PATH = "/v1/models"
MAX_REQUEST_BYTES = 1 << 20
DEFAULT_PORT = 8790

# Confidence rules a package may declare for its answers (SPEC.md, "Answers").
CONFIDENCE_RULES = ("max-probability", "entropy", "typesafe", "typesafe-ordinal", "none")


def parse_standard(value: object) -> tuple[int, int]:
    """(major, minor) of a "noulxp/X.Y" string (or an older "odxp/X.Y"), or ValueError."""
    if not isinstance(value, str) or not value.startswith((PREFIX, LEGACY_PREFIX)):
        raise ValueError(f"not a NoulXP standard version: {value!r}")
    try:
        major, minor = (int(x) for x in value.split("/", 1)[1].split("."))
    except ValueError:
        raise ValueError(f"not a NoulXP standard version: {value!r}") from None
    return major, minor


def canonical(value: object) -> str:
    """The version as NoulXP names it: "odxp/0.2" is "noulxp/0.2"."""
    major, minor = parse_standard(value)
    return f"{PREFIX}{major}.{minor}"


def same_version(a: object, b: object) -> bool:
    """Whether two declarations name the same version, under either name."""
    try:
        return canonical(a) == canonical(b)
    except ValueError:
        return a == b


def supported(value: object) -> bool:
    """Whether this implementation can run a package declaring `value`.

    Before 1.0 every minor version may change the format, so an engine runs
    exactly the versions it implements, and never a newer one.
    """
    return value in VERSIONS or value in LEGACY_VERSIONS


def at_least(value: object, version: str) -> bool:
    """Whether the declared standard `value` is `version` or newer."""
    try:
        return parse_standard(value) >= parse_standard(version)
    except ValueError:
        return False
