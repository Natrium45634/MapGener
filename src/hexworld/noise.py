"""Seeded gradient noise evaluated at arbitrary world coordinates.

Because the noise is a pure function of position (no grid state), geology that
uses it is independent of the hex resolution that later samples it.
"""

from __future__ import annotations

import numpy as np

from .rng import Stream

_ANGLES = np.arange(16) * (2.0 * np.pi / 16.0)
_GRAD = np.round(np.stack([np.cos(_ANGLES), np.sin(_ANGLES)], axis=1), 12)
_NORM = 1.0 / 0.7071067811865476  # brings classic 2-D Perlin to roughly [-1, 1]


class Perlin:
    def __init__(self, stream: Stream):
        perm = stream.permutation(256)
        self._perm = np.concatenate([perm, perm]).astype(np.int64)
        self._gidx = (stream.raw(256) % np.uint64(16)).astype(np.int64)

    def __call__(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        xf = np.floor(x)
        yf = np.floor(y)
        xi = xf.astype(np.int64) & 255
        yi = yf.astype(np.int64) & 255
        dx = x - xf
        dy = y - yf
        u = dx * dx * dx * (dx * (dx * 6.0 - 15.0) + 10.0)
        v = dy * dy * dy * (dy * (dy * 6.0 - 15.0) + 10.0)
        p = self._perm
        g = self._gidx

        def corner(ix, iy, fx, fy):
            grad = _GRAD[g[p[p[ix] + iy] & 255]]
            return grad[..., 0] * fx + grad[..., 1] * fy

        xi1 = (xi + 1) & 255
        yi1 = (yi + 1) & 255
        n00 = corner(xi, yi, dx, dy)
        n10 = corner(xi1, yi, dx - 1.0, dy)
        n01 = corner(xi, yi1, dx, dy - 1.0)
        n11 = corner(xi1, yi1, dx - 1.0, dy - 1.0)
        nx0 = n00 + u * (n10 - n00)
        nx1 = n01 + u * (n11 - n01)
        return (nx0 + v * (nx1 - nx0)) * _NORM


class FractalNoise:
    """Sum of Perlin octaves; ``amplitudes[k]`` at ``wavelengths_km[k]``."""

    def __init__(self, stream: Stream, wavelengths_km, amplitudes):
        self.wavelengths = [float(w) for w in wavelengths_km]
        self.amplitudes = [float(a) for a in amplitudes]
        self._octaves = []
        for k in range(len(self.wavelengths)):
            sub = stream.child(f"octave{k}")
            perlin = Perlin(sub)
            ox, oy = sub.uniform(0.0, 256.0, 2)
            self._octaves.append((perlin, float(ox), float(oy)))

    def __call__(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        out = np.zeros(np.broadcast(x, y).shape)
        for (perlin, ox, oy), wl, amp in zip(
            self._octaves, self.wavelengths, self.amplitudes, strict=True
        ):
            if amp != 0.0:
                out += amp * perlin(x / wl + ox, y / wl + oy)
        return out


class DomainWarp:
    """Smooth displacement field ``(x, y) -> (x + wx, y + wy)`` in km.

    ``strength`` is the sum over octaves of amplitude / wavelength. While the
    displacement gradient stays below 1 the map is one-to-one, so a connected
    region stays connected after warping; that is how plate connectivity is
    obtained by construction rather than by post-hoc repair. The default
    strengths stay in that regime in practice, and the ``validate`` stage
    checks the outcome on every run instead of trusting the bound.
    """

    def __init__(self, stream: Stream, wavelength_km: float, octaves: int, strength: float):
        self.wavelength_km = float(wavelength_km)
        self.octaves = int(octaves)
        self.strength = float(strength)
        wavelengths = [self.wavelength_km / (2.0**k) for k in range(self.octaves)]
        # Slightly favour long waves: weights ~ 0.85^k, normalised to `strength`.
        weights = np.array([0.85**k for k in range(self.octaves)])
        weights = weights / weights.sum() * self.strength if self.octaves else weights
        amplitudes = [w * wl for w, wl in zip(weights, wavelengths, strict=True)]
        self._nx = FractalNoise(stream.child("x"), wavelengths, amplitudes)
        self._ny = FractalNoise(stream.child("y"), wavelengths, amplitudes)

    def __call__(self, x, y):
        if self.strength == 0.0 or self.octaves == 0:
            return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        return x + self._nx(x, y), y + self._ny(x, y)

    def to_dict(self) -> dict:
        return {
            "wavelength_km": self.wavelength_km,
            "octaves": self.octaves,
            "strength": self.strength,
        }
