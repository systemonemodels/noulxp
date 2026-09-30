"""OpenDXP: a portable runtime standard for calibrated single-pass decision models.

This package is the reference implementation: the runtimes that run any
conforming package without model-specific code, the converters that make
packages from the models' own checkpoints, and the conformance tools.

    import opendxp
    model = opendxp.load("path/to/package")          # device chosen from what is available
    model.predict(state, questions)                 # {"answers": {...}, "usage": {...}}
"""

from __future__ import annotations

from typing import Any

__version__ = "0.3.1"


def load(package_dir: Any, **options: Any) -> Any:
    """Open a package and start the reference runtime for its profile."""
    from opendxp.runtime import load as _load

    return _load(package_dir, **options)


__all__ = ["__version__", "load"]
