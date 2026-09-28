"""What every converter does: place files, hash them, write the manifest."""

from __future__ import annotations

import json
import os
import shutil
import struct
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opendxp import __version__
from opendxp.package import MANIFEST, sha256_file
from opendxp.spec import BASE


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
        "source": {**source, "converted_by": f"opendxp {__version__}", "converted_at": now()},
    }


def write_manifest(out_dir: Path, data: dict[str, Any]) -> Path:
    path = out_dir / MANIFEST
    write_json(path, data)
    return path
