"""The opendxp command: noulxp's, with a note to switch."""

from __future__ import annotations

import sys


def main() -> int:
    print("opendxp is now noulxp: run `noulxp` (pip install noulxp)", file=sys.stderr)
    from noulxp.cli import main as noulxp

    return noulxp()
