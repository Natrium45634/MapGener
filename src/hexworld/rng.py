"""Deterministic, named random streams derived from a case-sensitive text seed.

Scheme (``PRNG_VERSION``): ``SHA-256("hexworld|<seed>|<stream>")`` -> eight
little-endian uint32 words -> ``numpy.random.SeedSequence`` -> ``PCG64``.
Only *raw* 64-bit outputs of the bit generator are consumed (NumPy guarantees
bit-generator stream stability), and all distributions are built here, so the
results do not depend on ``numpy.random.Generator`` implementation details,
Python's salted ``hash()`` or any hidden global RNG.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

_SEED_RE = re.compile(r"[0-9A-Za-z]+")
_TWO_POW_53_INV = 1.0 / 9007199254740992.0


def validate_seed(seed: str) -> str:
    """Return ``seed`` unchanged if it is a non-empty ``[0-9A-Za-z]+`` string."""
    if not isinstance(seed, str) or not _SEED_RE.fullmatch(seed):
        raise ValueError(
            "Seed must be a non-empty string of ASCII digits and Latin letters "
            "(0-9, A-Z, a-z); it is case sensitive."
        )
    return seed


def stream_digest(seed: str, stream: str) -> bytes:
    return hashlib.sha256(f"hexworld|{seed}|{stream}".encode("ascii")).digest()


class Stream:
    """Independent random stream identified by ``(seed, name)``."""

    def __init__(self, seed: str, name: str):
        validate_seed(seed)
        self.seed = seed
        self.name = name
        words = np.frombuffer(stream_digest(seed, name), dtype="<u4")
        self._bg = np.random.PCG64(np.random.SeedSequence([int(w) for w in words]))

    def child(self, name: str) -> Stream:
        return Stream(self.seed, f"{self.name}/{name}")

    def raw(self, n: int) -> np.ndarray:
        return self._bg.random_raw(int(n)).astype(np.uint64, copy=False)

    def random(self, n: int | None = None):
        """Uniform float64 in [0, 1) with 53 random bits."""
        if n is None:
            return float(self.random(1)[0])
        return (self.raw(n) >> np.uint64(11)).astype(np.float64) * _TWO_POW_53_INV

    def uniform(self, lo: float, hi: float, n: int | None = None):
        return lo + (hi - lo) * self.random(n)

    def integer(self, n: int) -> int:
        """Integer in [0, n)."""
        return min(int(self.random() * n), n - 1)

    def normal(self, n: int | None = None):
        """Standard normal via Box-Muller."""
        m = 1 if n is None else int(n)
        u1 = 1.0 - self.random(m)
        u2 = self.random(m)
        z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
        return float(z[0]) if n is None else z

    def permutation(self, n: int) -> np.ndarray:
        """Fisher-Yates shuffle of ``arange(n)``."""
        perm = np.arange(n, dtype=np.int64)
        u = self.random(n)
        for i in range(n - 1, 0, -1):
            j = min(int(u[i] * (i + 1)), i)
            perm[i], perm[j] = perm[j], perm[i]
        return perm
