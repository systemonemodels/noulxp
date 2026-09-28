"""Rotations (SPEC.md 6.8): asking one question with its options in several orders.

A letter-readout model prefers some positions over others. Showing the options
in every cyclic shift gives every option every position; the answers of the
shifts are then combined per option. Optionally a label prior, estimated
without labels from content-free probes, is divided out of each shift first.

The math is AnyJev's L0 (github.com/nokia-applied-research/AnyJev, anyjev 0.2.0,
Apache-2.0: calibrate/permute.py and calibrate/contextual.py): the same shifts,
the same clipping constants, the same order of operations.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Clipping before a logarithm, and before and after dividing by a prior.
LOG_EPS = 1e-12
PRIOR_EPS = 1e-8

COMBINE = ("logmean", "mean")


def cyclic_shifts(base: Sequence[int]) -> list[list[int]]:
    """Every cyclic shift of a listing: shift s shows base[(j + s) % k] at position j."""
    k = len(base)
    return [[base[(j + s) % k] for j in range(k)] for s in range(k)]


def listing(order: str, texts: Sequence[str]) -> list[int]:
    """The listing the shifts turn: the request's order, or the options sorted by text."""
    if order == "request":
        return list(range(len(texts)))
    if order == "canonical":
        return sorted(range(len(texts)), key=lambda i: (texts[i], i))
    raise ValueError(f"unknown rotation order {order!r}")


def prior_from_probes(p_probes: np.ndarray) -> np.ndarray:
    """[..., C, K] distributions over K positions for C probes -> the mean prior [..., K]."""
    prior = np.asarray(p_probes, dtype=np.float64).mean(axis=-2)
    prior = np.clip(prior, PRIOR_EPS, None)
    out: np.ndarray = prior / prior.sum(axis=-1, keepdims=True)
    return out


def divide_prior(p_pos: np.ndarray, prior: np.ndarray, strength: float) -> np.ndarray:
    """normalise(p / prior**strength), both [..., K] over positions."""
    divisor = np.clip(np.power(np.asarray(prior, dtype=np.float64), strength), PRIOR_EPS, None)
    p = np.clip(np.asarray(p_pos, dtype=np.float64) / divisor, PRIOR_EPS, None)
    out: np.ndarray = p / p.sum(axis=-1, keepdims=True)
    return out


def combine(p_pos: np.ndarray, perms: Sequence[Sequence[int]], how: str = "logmean") -> np.ndarray:
    """[P, K] distributions by position -> [K] by option (perm[j] is shown at position j).

    "logmean": the geometric mean, renormalised: if position adds a bias in logit
    space, it cancels exactly. "mean": the arithmetic mean.
    """
    p_pos = np.asarray(p_pos, dtype=np.float64)
    shifts, k = p_pos.shape
    per_option = np.zeros((shifts, k))
    for s, perm in enumerate(perms):
        for j, i in enumerate(perm):
            per_option[s, i] = p_pos[s, j]
    if how == "logmean":
        z = np.log(np.clip(per_option, LOG_EPS, None)).mean(axis=0)
        out = np.exp(z - z.max())
    elif how == "mean":
        out = per_option.mean(axis=0)
    else:
        raise ValueError(f"combine must be one of {COMBINE}")
    result: np.ndarray = out / out.sum()
    return result
