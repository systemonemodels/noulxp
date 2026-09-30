"""What a server holds, and how it answers: shared by the HTTP and MCP bindings.

A server loads packages with the reference runtime, describes each one for
discovery (SPEC.md 11.3), and answers a request with one of them. The runtimes
are not safe to call from two threads at once (a llama.cpp context is one
object), so each model has one thread that calls it: requests that arrive while
it reads others wait, and are read together in its next pass (`Batcher`).
"""

from __future__ import annotations

import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from noulxp import schemas, spec
from noulxp.errors import RequestError
from noulxp.package import Package, open_package


class ServingError(Exception):
    """A request the server turns down, with the error type SPEC.md 11.4 names."""

    def __init__(self, kind: str, message: str, status: int) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.status = status


class Batcher:
    """A model's runtime, read by one request at a time or by several together.

    A request that finds the model idle and nobody waiting is read at once, in
    its caller's thread, as `predict` reads it. One that finds it busy waits in a
    queue, and a thread of the batcher's own takes the oldest and every other one
    waiting, up to `limit`, and reads them together (`predict_many`: on a GPU,
    rows of similar length from all of them share a pass). Nothing waits on
    purpose, so passes grow with the load. Each request gets its own answer, or
    its own error.
    """

    def __init__(self, runtime: Any, limit: int, name: str = "model") -> None:
        self.runtime = runtime
        self.limit = max(1, int(limit))
        self.queue: queue.SimpleQueue[tuple[Any, Any, Future[Any]] | None] = queue.SimpleQueue()
        self.closed = False
        self._closing = threading.Lock()
        self.reading = threading.Lock()  # held while the runtime reads, by a caller or the thread
        self.thread = threading.Thread(target=self._work, name=f"noulxp {name}", daemon=True)
        self.thread.start()

    def answer(self, state: Any, questions: Any) -> Any:
        """The request's answer: read here if the model is idle and nobody waits, else queued."""
        if self.queue.empty() and self.reading.acquire(blocking=False):
            try:
                if self.closed:
                    raise ServingError("model_not_ready", "the server is stopping", 503)
                return self.runtime.predict(state, questions)
            finally:
                self.reading.release()
        return self.submit(state, questions).result()

    def submit(self, state: Any, questions: Any) -> Future[Any]:
        future: Future[Any] = Future()
        with self._closing:
            if not self.closed:
                self.queue.put((state, questions, future))
                return future
        future.set_exception(ServingError("model_not_ready", "the server is stopping", 503))
        return future

    def _work(self) -> None:
        while True:
            first = self.queue.get()
            if first is None:
                return
            with self.reading:  # the ones that queue while a caller reads join this pass
                batch = [first]
                stop = False
                while len(batch) < self.limit:
                    try:
                        waiting = self.queue.get_nowait()
                    except queue.Empty:
                        break
                    if waiting is None:
                        stop = True
                        break
                    batch.append(waiting)
                try:
                    self._answer(batch)
                except Exception as exc:  # never lose the thread: the requests get the error
                    for _, _, future in batch:
                        if not future.done():
                            future.set_exception(exc)
            if stop:
                return

    def _answer(self, batch: list[tuple[Any, Any, Future[Any]]]) -> None:
        items = [(state, questions) for state, questions, _ in batch]
        results: list[Any]
        if len(items) == 1:  # alone, a request is read as `predict` reads it: in one pass
            results = [self._one(*items[0])]
        else:
            try:
                results = self.runtime.predict_many(items)
            except Exception:
                # The pass failed: read them one by one, so the error is the request's own.
                results = [self._one(state, questions) for state, questions in items]
        for (_, _, future), result in zip(batch, results, strict=True):
            if isinstance(result, BaseException):
                future.set_exception(result)
            else:
                future.set_result(result)

    def _one(self, state: Any, questions: Any) -> Any:
        try:
            return self.runtime.predict(state, questions)
        except Exception as exc:
            return exc

    def close(self) -> None:
        """Stop taking requests, answer the ones already waiting, and end the thread."""
        with self._closing:
            if self.closed:
                return
            self.closed = True
            self.queue.put(None)  # after every request taken, none after it
        self.thread.join()
        with self.reading:  # a caller's read under way ends before the runtime is closed
            pass


@dataclass
class ServedModel:
    """One package, loaded and ready to answer.

    With `batch` above 1, requests are read by a `Batcher`, up to that many in a
    pass; with 1, one at a time, in the calling thread.
    """

    id: str
    package: Package
    runtime: Any
    conformance: dict[str, Any] | None = None
    batch: int = 1
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    batcher: Batcher | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.batch > 1:
            self.batcher = Batcher(self.runtime, self.batch, self.id)

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
            "calibration": getattr(self.runtime, "calibration_replaced", None),
        }

    def answer(self, state: Any, questions: Any) -> dict[str, Any]:
        """The model's answer to one request: its answers and the tokens it read."""
        result: dict[str, Any]
        if self.batcher is not None:
            result = self.batcher.answer(state, questions)
        else:
            with self.lock:
                result = self.runtime.predict(state, questions)
        return {"answers": result["answers"], "usage": result["usage"]}

    def close(self) -> None:
        if self.batcher is not None:
            self.batcher.close()
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
    calibration: str | Path | None = None,
    batch: int = 1,
    batch_rows: int | None = None,
) -> list[ServedModel]:
    """Open and load each package; with `check`, run its conformance file first.

    `calibration` is a calibration.json to answer with in place of the package's
    (SPEC.md 7.1), for a server of one package; the check is of the package as it is.
    `batch` is how many waiting requests a model reads in one pass (1: one at a
    time), and `batch_rows` how many rows a causal-letters package decodes together."""
    from noulxp.runtime import load

    if calibration is not None and len(paths) != 1:
        raise ValueError("a calibration file is for one package: serve the others apart")
    models: list[ServedModel] = []
    for path in paths:
        package = open_package(path)
        report = None
        if check:
            from noulxp.conformance import check as run_check

            log(f"checking {package.name} against its conformance file ...")
            report = summarise(run_check(Path(path), device="cpu", threads=threads, log=log))
        log(f"loading {package.name} ({package.profile}) ...")
        options: dict[str, Any] = {}
        if batch_rows and batch_rows > 1 and package.profile == "causal-letters":
            options["batch_rows"] = batch_rows
        runtime = load(
            package,
            device=device,
            threads=threads,
            precision=precision,
            calibration=calibration,
            **options,
        )
        if calibration is not None:
            log(f"answering at {calibration} in place of {package.name}'s calibration")
        models.append(ServedModel(package.name, package, runtime, conformance=report, batch=batch))
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
