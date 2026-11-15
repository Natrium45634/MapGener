"""ctypes binding to the PMGE shared library (C ABI version 1).

HexWorld works in km, PMGE in metres: every coordinate and length is converted at
this boundary and nowhere else. The library is found, in this order, through
``HEXWORLD_PMGE_LIBRARY``, the ``pmge/build`` directory of this repository, and the
system library path.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from pathlib import Path

import numpy as np

ABI_VERSION = 1
KM = 1000.0

_dbl = ctypes.c_double
_i32 = ctypes.c_int32
_i64 = ctypes.c_int64
_P = ctypes.POINTER


class Request(ctypes.Structure):
    _fields_ = [
        ("seed", ctypes.c_uint64),
        ("footprint_type", _i32),
        ("center_x", _dbl),
        ("center_y", _dbl),
        ("radius_m", _dbl),
        ("box_min_x", _dbl),
        ("box_min_y", _dbl),
        ("box_max_x", _dbl),
        ("box_max_y", _dbl),
        ("base_m", _dbl),
        ("min_m", _dbl),
        ("max_m", _dbl),
        ("peak_count", _i32),
        ("peak_spacing_m", _dbl),
        ("peak_type_weights", _dbl * 8),
        ("ridge_half_width_m", _dbl),
        ("spur_spacing_m", _dbl),
        ("spur_length_m", _dbl),
        ("spur_half_width_m", _dbl),
        ("foothill_width_m", _dbl),
        ("foothill_profile", _i32),
        ("maturity", _dbl),
        ("valley_depth", _dbl),
        ("detail", _dbl),
    ]


class Sample(ctypes.Structure):
    _fields_ = [
        ("height", _dbl),
        ("relief", _dbl),
        ("structure", _dbl),
        ("valley", _dbl),
        ("ridge_distance", _dbl),
        ("system", _i32),
        ("class_id", _i32),
    ]


SAMPLE_DTYPE = np.dtype(
    [
        ("height", "f8"),
        ("relief", "f8"),
        ("structure", "f8"),
        ("valley", "f8"),
        ("ridge_distance", "f8"),
        ("system", "i4"),
        ("class_id", "i4"),
    ],
    align=True,
)


def _candidates() -> list[str]:
    names = {"win32": ["pmge.dll"], "darwin": ["libpmge.dylib"]}.get(sys.platform, ["libpmge.so"])
    out = []
    env = os.environ.get("HEXWORLD_PMGE_LIBRARY")
    if env:
        out.append(env)
    repo = Path(__file__).resolve().parents[3] / "pmge"
    for sub in ("build", "build/Release", "build/Debug"):
        out += [str(repo / sub / n) for n in names]
    found = ctypes.util.find_library("pmge")
    if found:
        out.append(found)
    return out


def find_library() -> str | None:
    for path in _candidates():
        if os.path.isfile(path) or (path and not os.path.dirname(path)):
            return path
    return None


class Library:
    def __init__(self, path: str):
        lib = ctypes.CDLL(path)
        self.path = path
        lib.pmge_abi_version.restype = ctypes.c_int
        lib.pmge_version.restype = ctypes.c_char_p
        lib.pmge_last_error.restype = ctypes.c_char_p
        lib.pmge_request_init.argtypes = [_P(Request)]
        lib.pmge_plan_create_external.restype = ctypes.c_void_p
        lib.pmge_plan_create_external.argtypes = [
            _P(Request),
            _i32,
            _P(_i32),
            _P(_dbl),
            _P(_dbl),
            _P(_dbl),
            _P(_i32),
            _P(_i32),
            _P(_dbl),
            _i32,
            _P(_i32),
            _P(_dbl),
            _P(_dbl),
            _P(_dbl),
            _dbl,
            _dbl,
            _dbl,
            _i32,
            _i32,
            _P(ctypes.c_float),
            _P(ctypes.c_float),
            _dbl,
        ]
        lib.pmge_plan_destroy.argtypes = [ctypes.c_void_p]
        lib.pmge_sample_batch.restype = ctypes.c_int
        lib.pmge_sample_batch.argtypes = [
            ctypes.c_void_p,
            _P(_dbl),
            _P(_dbl),
            _i64,
            ctypes.c_void_p,
        ]
        self.lib = lib
        self.abi = int(lib.pmge_abi_version())
        self.version = lib.pmge_version().decode()

    def error(self) -> str:
        return (self.lib.pmge_last_error() or b"").decode()


def _arr(values, ctype, dtype):
    a = np.ascontiguousarray(values, dtype=dtype)
    return a, a.ctypes.data_as(_P(ctype))


class ExternalPlan:
    """A PMGE plan built from HexWorld's own mountain structure (ridges, valleys, relief)."""

    def __init__(
        self,
        library: Library,
        seed: int,
        detail: float,
        ridges,
        valleys,
        grid,
        amplitude,
        gain,
        mean_offset: float,
    ):
        self.library = library
        lib = library.lib
        req = Request()
        lib.pmge_request_init(ctypes.byref(req))
        req.seed = seed & 0xFFFFFFFFFFFFFFFF
        req.detail = detail
        keep = []

        def ptr(values, ctype, dtype):
            a, p = _arr(values, ctype, dtype)
            keep.append(a)
            return p

        r_off, r_xy, r_crest, r_hw, r_rank, r_sys, r_sharp = [0], [], [], [], [], [], []
        for r in ridges:
            r_xy.append(np.asarray(r["points"], dtype=np.float64) * KM)
            r_crest.append(np.asarray(r["crest"], dtype=np.float64))
            r_off.append(r_off[-1] + len(r["points"]))
            r_hw.append(r["half_width_km"] * KM)
            r_rank.append(r["rank"])
            r_sys.append(r["system"])
            r_sharp.append(r["sharpness"])
        v_off, v_xy, v_depth, v_hw = [0], [], [], []
        for v in valleys:
            v_xy.append(np.asarray(v["points"], dtype=np.float64) * KM)
            v_depth.append(np.asarray(v["depth"], dtype=np.float64))
            v_off.append(v_off[-1] + len(v["points"]))
            v_hw.append(v["half_width_km"] * KM)

        def cat(parts, width=None):
            if parts:
                return np.concatenate(parts)
            return np.zeros((0, width) if width else 0)

        amp = np.ascontiguousarray(amplitude, dtype=np.float32)
        gn = np.ascontiguousarray(gain, dtype=np.float32)
        keep += [amp, gn]
        handle = lib.pmge_plan_create_external(
            ctypes.byref(req),
            len(ridges),
            ptr(r_off, _i32, np.int32),
            ptr(cat(r_xy, 2).ravel(), _dbl, np.float64),
            ptr(cat(r_crest), _dbl, np.float64),
            ptr(r_hw, _dbl, np.float64),
            ptr(r_rank, _i32, np.int32),
            ptr(r_sys, _i32, np.int32),
            ptr(r_sharp, _dbl, np.float64),
            len(valleys),
            ptr(v_off, _i32, np.int32),
            ptr(cat(v_xy, 2).ravel(), _dbl, np.float64),
            ptr(cat(v_depth), _dbl, np.float64),
            ptr(v_hw, _dbl, np.float64),
            grid["x0_km"] * KM,
            grid["y0_km"] * KM,
            grid["cell_km"] * KM,
            int(grid["nx"]),
            int(grid["ny"]),
            amp.ctypes.data_as(_P(ctypes.c_float)),
            gn.ctypes.data_as(_P(ctypes.c_float)),
            float(mean_offset),
        )
        if not handle:
            raise RuntimeError(f"PMGE rejected the plan: {library.error()}")
        self.handle = handle

    def sample(self, x_km, y_km) -> np.ndarray:
        x = np.ascontiguousarray(np.ravel(x_km), dtype=np.float64) * KM
        y = np.ascontiguousarray(np.ravel(y_km), dtype=np.float64) * KM
        out = np.zeros(x.size, dtype=SAMPLE_DTYPE)
        rc = self.library.lib.pmge_sample_batch(
            self.handle,
            x.ctypes.data_as(_P(_dbl)),
            y.ctypes.data_as(_P(_dbl)),
            x.size,
            out.ctypes.data,
        )
        if rc != 0:
            raise RuntimeError(f"PMGE sampling failed: {self.library.error()}")
        return out

    def __del__(self):
        handle = getattr(self, "handle", None)
        if handle:
            self.library.lib.pmge_plan_destroy(handle)
            self.handle = None
