"""A NoulXP package on disk: a directory with noulxp.json and the files it names.

Nothing in a package is executed. The manifest names every file with its
SHA-256, and a runtime reads only those files, and only from inside the
package directory.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from noulxp.errors import PackageError
from noulxp.spec import CONFIDENCE_RULES, PROFILES, VERSIONS, supported

MANIFEST = "noulxp.json"
# The manifest's name in packages made before the rename (OpenDXP, until 0.3.1).
LEGACY_MANIFESTS = ("odxp.json",)
# The files each profile needs, by manifest key.
REQUIRED = {
    "encoder-markers": ("weights", "tokenizer", "template", "calibration"),
    "causal-letters": ("weights", "tokenizer", "prompt", "calibration"),
}
WEIGHT_FORMATS = {"encoder-markers": "onnx", "causal-letters": "gguf"}


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _safe(relative: str) -> bool:
    parts = PurePosixPath(relative).parts
    return (
        bool(parts)
        and not relative.startswith("/")
        and ".." not in parts
        and "\\" not in relative
        and ":" not in relative
    )


def find_manifest(root: Path) -> Path | None:
    """The package's manifest: noulxp.json, or odxp.json in a package made before the rename."""
    for name in (MANIFEST, *LEGACY_MANIFESTS):
        if (root / name).is_file():
            return root / name
    return None


@dataclass
class Package:
    root: Path
    manifest: dict[str, Any]
    manifest_name: str = MANIFEST

    @property
    def name(self) -> str:
        return str(self.manifest.get("name", self.root.name))

    @property
    def profile(self) -> str:
        return str(self.manifest["profile"])

    @property
    def limits(self) -> dict[str, int]:
        return dict(self.manifest.get("limits") or {})

    @property
    def confidence_rules(self) -> dict[str, str]:
        rules = dict((self.manifest.get("answers") or {}).get("confidence") or {})
        for qtype, rule in rules.items():
            if rule not in CONFIDENCE_RULES:
                raise PackageError(f"unknown confidence rule {rule!r} for {qtype}")
        return rules

    def entry(self, key: str) -> dict[str, Any]:
        value = self.manifest.get(key)
        if not isinstance(value, dict) or not isinstance(value.get("path"), str):
            raise PackageError(f"{self.manifest_name}: {key} must be an object with a path")
        return value

    def resolve(self, relative: str) -> Path:
        if not _safe(relative):
            raise PackageError(f"not a path inside the package: {relative!r}")
        return self.root / relative

    def file(self, key: str) -> Path:
        return self.resolve(self.entry(key)["path"])

    def read_json(self, key: str) -> dict[str, Any]:
        path = self.file(key)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PackageError(f"cannot read {path.name}: {exc}") from exc
        if not isinstance(data, dict):
            raise PackageError(f"{path.name} is not a JSON object")
        return data

    def files(self) -> list[tuple[str, dict[str, Any]]]:
        """Every (manifest key, file entry) the package lists, weights data included."""
        out: list[tuple[str, dict[str, Any]]] = []
        for key in ("weights", "tokenizer", "template", "prompt", "calibration", "conformance"):
            value = self.manifest.get(key)
            if isinstance(value, dict) and "path" in value:
                out.append((key, value))
                for extra in value.get("data") or []:
                    out.append((f"{key}.data", extra))
        return out

    def verify(self, *, hashes: bool = True) -> list[str]:
        """Problems with the files: missing, outside the package, or not matching their hash."""
        problems = []
        for key, value in self.files():
            relative = value.get("path", "")
            if not _safe(relative):
                problems.append(f"{key}: {relative!r} is not a path inside the package")
                continue
            path = self.root / relative
            if not path.is_file():
                problems.append(f"{key}: {relative} is missing")
                continue
            if hashes and value.get("sha256") and sha256_file(path) != value["sha256"]:
                problems.append(f"{key}: {relative} does not match its sha256")
        return problems


def open_package(path: str | Path) -> Package:
    root = Path(path)
    manifest_path = find_manifest(root)
    if manifest_path is None:
        raise PackageError(f"{root} has no {MANIFEST}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise PackageError(f"{manifest_path.name} is not JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise PackageError(f"{manifest_path.name} is not an object")
    standard = manifest.get("standard")
    if not supported(standard):
        raise PackageError(f"this engine runs {' and '.join(VERSIONS)} packages, not {standard!r}")
    profile = manifest.get("profile")
    if profile not in PROFILES:
        raise PackageError(f"unknown profile {profile!r}")
    package = Package(root, manifest, manifest_path.name)
    for key in REQUIRED[profile]:
        if not package.file(key).is_file():
            raise PackageError(f"{key}: {package.entry(key)['path']} is missing")
    fmt = package.entry("weights").get("format", WEIGHT_FORMATS[profile])
    if fmt != WEIGHT_FORMATS[profile]:
        raise PackageError(f"{profile} packages carry {WEIGHT_FORMATS[profile]} weights, not {fmt}")
    return package
