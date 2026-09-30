"""OpenDXP is now NoulXP: this module is noulxp under its old name.

`import opendxp` gives noulxp's top-level names, and `import opendxp.package`
(or any other submodule) gives the noulxp module itself, the same object, so
code written against opendxp keeps working. Use noulxp directly.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
import warnings
from types import ModuleType
from typing import Any

from noulxp import *  # noqa: F403
from noulxp import __all__, __version__  # noqa: F401

warnings.warn(
    "opendxp is now noulxp: pip install noulxp, and import noulxp instead of opendxp",
    DeprecationWarning,
    stacklevel=2,
)


class _Renamed(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Finds opendxp.<name> as the module noulxp.<name>."""

    def __init__(self) -> None:
        self._specs: dict[str, Any] = {}

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if not fullname.startswith("opendxp.") or fullname == "opendxp.__main__":
            return None
        return importlib.machinery.ModuleSpec(fullname, self)

    def create_module(self, spec: Any) -> ModuleType:
        module = importlib.import_module("noulxp" + spec.name.removeprefix("opendxp"))
        self._specs[spec.name] = module.__spec__
        return module

    def exec_module(self, module: ModuleType) -> None:
        # The import system gave the noulxp module an opendxp spec; give its own back.
        module.__spec__ = self._specs.pop(module.__spec__.name)


sys.meta_path.insert(0, _Renamed())
