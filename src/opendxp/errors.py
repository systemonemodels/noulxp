"""Errors, split by whose fault they are."""

from __future__ import annotations


class OpenDXPError(Exception):
    """Anything OpenDXP refuses."""


class PackageError(OpenDXPError):
    """The package is not a valid OpenDXP package, or this engine cannot run it."""


class RequestError(OpenDXPError, ValueError):
    """This package cannot answer this request (invalid, over a limit, or a reserved token).

    A conformance case may expect one: when the model's own code refuses a
    request, a faithful runtime refuses it too.
    """


class BackendUnavailable(OpenDXPError):
    """A library or device the runtime needs is not installed on this machine."""
