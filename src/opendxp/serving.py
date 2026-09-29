"""What a server holds, and how it answers: shared by the HTTP and MCP bindings.

A server loads packages with the reference runtime, describes each one for
discovery (SPEC.md 11.3), and answers a request with one of them. The runtimes
are not safe to call from two threads at once (a llama.cpp context is one
object), so each model answers one request at a time.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from opendxp import schemas, spec
from opendxp.errors import RequestError
from opendxp.package import Package, open_package


class ServingError(Exception):
    """A request the server turns down, with the error type SPEC.md 11.4 names."""

    def __init__(self, kind: str, message: str, status: int) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.status = status


@dataclass
class ServedModel:
    """One package, loaded and ready to answer."""

    id: str
    package: Package
    runtime: Any
    conformance: dict[str, Any] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def describe(self) -> dict[str, Any]:
        """The model's entry in a discovery answer (SPEC.md 11.3)."""
        manifest = self.package.manifest
        return {
            "id": self.id,
            "object": "model",
            "standard": manifest.get("standard", spec.STANDARD),
            "profile": self.package.profile,
            "question_types": list(spec.QUESTION_TYPES),
            "limits": self.package.limits,
            "conformance": self.conformance,
        }

    def answer(self, state: Any, questions: Any) -> dict[str, Any]:
        """The model's answer to one request: its answers and the tokens it read."""
        with self.lock:
            result: dict[str, Any] = self.runtime.predict(state, questions)
        return {"answers": result["answers"], "usage": result["usage"]}

    def close(self) -> None:
        close = getattr(self.runtime, "close", None)
        if close is not None:
            close()


def summarise(report: dict[str, Any]) -> dict[str, Any]:
    """A conformance report reduced to what discovery shows."""
    agreement = report.get("argmax_agreement") or {}
    return {
        "compatible": bool(report.get("compatible")),
        "cases": report.get("cases"),
        "cases_passed": report.get("cases_passed"),
        "max_abs_dp": report.get("max_abs_dp"),
        "decisions_equal": f"{agreement.get('agree')}/{agreement.get('total')}",
        "device": (report.get("runtime") or {}).get("device"),
    }


def load_models(
    paths: list[str | Path],
    *,
    device: str = "auto",
    threads: int | None = None,
    check: bool = False,
    log: Any = print,
    precision: str = "fast",
) -> list[ServedModel]:
    """Open and load each package; with `check`, run its conformance file first."""
    from opendxp.runtime import load

    models: list[ServedModel] = []
    for path in paths:
        package = open_package(path)
        report = None
        if check:
            from opendxp.conformance import check as run_check

            log(f"checking {package.name} against its conformance file ...")
            report = summarise(run_check(Path(path), device="cpu", threads=threads, log=log))
        log(f"loading {package.name} ({package.profile}) ...")
        runtime = load(package, device=device, threads=threads, precision=precision)
        models.append(ServedModel(package.name, package, runtime, conformance=report))
    ids = [m.id for m in models]
    if len(set(ids)) != len(ids):
        raise ValueError(f"two packages share a name: {sorted(ids)}")
    return models


def find(models: list[ServedModel], wanted: Any) -> ServedModel:
    """The model a request names, or the only one a server holds."""
    if wanted is None:
        if len(models) == 1:
            return models[0]
        raise ServingError(
            "invalid_request",
            f"this server holds {len(models)} models; name one in `model`: "
            + ", ".join(m.id for m in models),
            400,
        )
    if not isinstance(wanted, str):
        raise ServingError("invalid_request", "`model` must be a string", 400)
    for model in models:
        if model.id == wanted:
            return model
    raise ServingError("model_not_found", f"no model {wanted!r} on this server", 404)


def request_problems(request: Any) -> list[str]:
    """How a request breaks SPEC.md 3, as readable lines; empty when it is valid."""
    if not isinstance(request, dict):
        return ["a request must be a JSON object"]
    body = {k: v for k, v in request.items() if k != "model"}
    return schemas.errors(body, "request")


def decide(models: list[ServedModel], request: Any) -> dict[str, Any]:
    """Answer one request (SPEC.md 11.2): the model, its answers and the tokens read."""
    problems = request_problems(request)
    if problems:
        raise ServingError("invalid_request", "; ".join(problems[:5]), 400)
    model = find(models, request.get("model"))
    try:
        result = model.answer(request.get("state", ""), request["questions"])
    except RequestError as exc:
        raise ServingError("request_refused", str(exc), 422) from exc
    return {"model": model.id, **result}
