"""What every converter does: place files, hash them, write the manifest."""

from __future__ import annotations

import json
import os
import shutil
import struct
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from noulxp import __version__
from noulxp.errors import BackendUnavailable
from noulxp.package import MANIFEST, sha256_file
from noulxp.spec import BASE

# The encoder converters trace the model as transformers builds it. Older releases compute
# ModernBERT (Laya, Julia 1) differently: exported with 4.57.6, Julia 1 passed 7 of its 52
# conformance cases and Laya multilingual 2. The graph agrees with the model it was traced
# from, so the converter's own comparison with torch cannot see it; only the conformance
# file, recorded with the model's own code, does.
MIN_TRANSFORMERS = (5, 2)


def require_transformers() -> None:
    """Refuse to export an encoder with a transformers release older than MIN_TRANSFORMERS."""
    import importlib.metadata
    import re

    try:
        found = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as exc:
        raise BackendUnavailable(
            'the encoder converters need transformers: pip install "noulxp[export]"'
        ) from exc
    have = tuple(int(part) for part in re.findall(r"\d+", found)[: len(MIN_TRANSFORMERS)])
    if have < MIN_TRANSFORMERS:
        wanted = ".".join(map(str, MIN_TRANSFORMERS))
        raise BackendUnavailable(
            f"transformers {found} is installed and the encoder converters need {wanted} or "
            "later: older releases compute ModernBERT differently, and the package would not "
            f'give the model\'s own answers (pip install "transformers>={wanted}")'
        )


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def place(source: Path, target: Path, mode: str = "link") -> str:
    """Put `source` at `target`: a hard link when asked and possible, else a copy.

    A hard link is a real directory entry: the package stays whole if the
    checkpoint is deleted, and a multi-gigabyte weights file costs no space twice.
    Returns how the file was placed.
    """
    source = Path(os.path.realpath(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if mode == "link":
        try:
            os.link(source, target)
            return "hardlink"
        except OSError:
            pass
    shutil.copyfile(source, target)
    return "copy"


def entry(root: Path, relative: str, **extra: Any) -> dict[str, Any]:
    return {"path": relative, "sha256": sha256_file(root / relative), **extra}


def safetensors_index(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    """The tensors of a safetensors file (name -> dtype, shape, byte range) and where data starts.

    The format is an 8-byte little-endian header length, a JSON header, then
    every tensor as one contiguous little-endian byte range, so any tensor can be
    read, or referenced as ONNX external data, by its offset.
    """
    with open(path, "rb") as fh:
        (size,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(size))
    header.pop("__metadata__", None)
    return header, 8 + size


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def manifest(
    *,
    name: str,
    profile: str,
    files: dict[str, dict[str, Any]],
    limits: dict[str, int],
    confidence: dict[str, str],
    source: dict[str, Any],
    standard: str = BASE,
) -> dict[str, Any]:
    return {
        "standard": standard,
        "name": name,
        "profile": profile,
        **files,
        "answers": {"confidence": confidence},
        "limits": limits,
        "source": {**source, "converted_by": f"noulxp {__version__}", "converted_at": now()},
    }


def write_manifest(out_dir: Path, data: dict[str, Any]) -> Path:
    path = out_dir / MANIFEST
    write_json(path, data)
    return path
