"""`noulxp validate`: a package checked without running it.

The manifest and each declarative file against its schema, then the same
files through the runtimes' own parsers (which check what a schema cannot:
placeholders, layout parts, label ranges), every conformance line, and the
SHA-256 of every file.
"""

from __future__ import annotations

from pathlib import Path

from noulxp.calibration import Calibration
from noulxp.conformance import coverage, read_jsonl
from noulxp.errors import NoulXPError
from noulxp.package import open_package
from noulxp.schemas import errors
from noulxp.spec import same_version


def validate_package(path: Path, *, hashes: bool = True) -> list[str]:
    try:
        package = open_package(path)
    except NoulXPError as exc:
        return [str(exc)]
    problems = [f"{package.manifest_name} {e}" for e in errors(package.manifest, "manifest")]
    declarative = "template" if package.profile == "encoder-markers" else "prompt"
    for key in (declarative, "calibration"):
        try:
            data = package.read_json(key)
        except NoulXPError as exc:
            problems.append(str(exc))
            continue
        problems += [f"{package.entry(key)['path']} {e}" for e in errors(data, key)]
        if isinstance(data, dict) and not same_version(
            data.get("standard"), package.manifest.get("standard")
        ):
            problems.append(
                f"{package.entry(key)['path']} declares {data.get('standard')!r}, "
                f"the manifest {package.manifest.get('standard')!r}"
            )
        try:
            if key == "template":
                from noulxp.profiles.encoder_markers import Template

                Template(data)
                if package.limits.get("max_tokens") != data["budgets"]["total"]:
                    problems.append("limits.max_tokens must equal the template's budgets.total")
            elif key == "prompt":
                from noulxp.profiles.causal_letters import Prompt

                Prompt(data)
            else:
                Calibration(data)
        except NoulXPError as exc:
            problems.append(str(exc))
    entry = package.manifest.get("conformance")
    if entry:
        cases = read_jsonl(package.resolve(entry["path"]))
        for case in cases:
            problems += [
                f"conformance {case.get('id')}: {e}" for e in errors(case, "conformance-case")
            ]
        if entry.get("cases") != len(cases):
            problems.append("conformance: the manifest's case count does not match the file")
        cov = coverage(cases)
        if not cov["ok"]:
            missing = [k for k, ok in cov["checks"].items() if not ok]
            problems.append(f"conformance coverage below the minimum: {', '.join(missing)}")
    else:
        problems.append("no conformance file: the package cannot be checked")
    problems += package.verify(hashes=hashes)
    return problems
