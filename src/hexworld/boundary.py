"""Regional boundary profiles (TZ 5.4.1 G1): what enters the simulation domain through its
open edges - air masses (temperature, humidity, wind) and ocean exchange.

A profile covers one or more sides of the simulation rectangle (``west``, ``east``,
``north``, ``south``) - a grouping of the boundary faces, not a replacement of the hex /
raster topology - and gives 12 monthly values. Two atmospheric forms, never both:

* ``prescribed_state``: the external column (T, q or RH, p); the inflow uses it as the
  upwind donor, the outflow uses the interior state (upwind scheme);
* ``prescribed_flux``: a moisture flux per metre of boundary (kg m^-1 s^-1, + inward).

Ocean sides exchange water with an external sea (``exchange_m3s`` equal in and out, at
``T_c``): a heat exchange without a net volume flux. A net prescribed inflow into a
closed basin is rejected (incompatible volume, T17). Units are checked; nothing is
replaced by zeros. The canonical JSON (sorted keys, fixed precision) hashes the same in
the GUI, the CLI and a saved project (T12).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field

SIDES = ("west", "east", "north", "south")
ATM_MODES = ("closed", "interior", "prescribed_state", "prescribed_flux")
OCN_MODES = ("closed", "exchange")
#: column water of the atmosphere per unit surface specific humidity (rho_air H_v), kg/m^2
COLUMN_PER_Q = 1.2 * 2200.0
EPS = 0.622


def e_sat_pa(t_c: float) -> float:
    return 611.2 * math.exp(17.67 * t_c / (t_c + 243.5))


def rh_to_q(rh: float, t_c: float, p_pa: float = 101325.0) -> float:
    """Specific humidity (kg/kg) from relative humidity (0..1), T (degC) and p (Pa) (T13)."""
    if not 0.0 <= rh <= 1.0:
        raise ValueError(f"relative humidity {rh} outside 0..1 (give a fraction, not %)")
    e = rh * e_sat_pa(t_c)
    if p_pa <= e:
        raise ValueError("pressure below the vapour pressure: invalid state")
    return EPS * e / (p_pa - (1.0 - EPS) * e)


def boundary_vapor_flux(length_m, w_ext, w_int, u_n):
    """Vapour flux through one boundary face, kg/s, + outward (N02): upwind donor."""
    w = w_ext if u_n < 0 else w_int
    return length_m * u_n * w


@dataclass
class BoundaryProfile:
    boundary_id: str
    sides: tuple
    atm_mode: str = "interior"
    t_air_c: tuple = ()  # 12 values
    q_kgkg: tuple = ()  # 12 values (or rh)
    rh: tuple = ()  # 12 values 0..1, converted to q with T and p
    p_pa: float = 101325.0
    moisture_flux_kg_m_s: tuple = ()  # 12 values, + inward
    ocn_mode: str = "closed"
    exchange_m3s: tuple = ()  # 12 values, per side group
    ocean_t_c: tuple = ()  # 12 values
    source: str = "manual"
    quality: str = "estimated"
    preset: str = ""
    extra: dict = field(default_factory=dict)

    def q_monthly(self):
        if self.q_kgkg:
            return list(self.q_kgkg)
        if self.rh:
            return [rh_to_q(r, t, self.p_pa) for r, t in zip(self.rh, self.t_air_c, strict=True)]
        return []

    def column_kg_m2(self):
        """External column water per month (the inflow donor of the column model)."""
        return [COLUMN_PER_Q * q for q in self.q_monthly()]


PRESETS = {
    "warm_ocean": dict(
        t_air_c=(26,) * 12,
        rh=(0.8,) * 12,
        atm_mode="prescribed_state",
        ocn_mode="exchange",
        ocean_t_c=(27,) * 12,
        exchange_m3s=(2.0e6,) * 12,
    ),
    "cold_ocean": dict(
        t_air_c=(6,) * 12,
        rh=(0.85,) * 12,
        atm_mode="prescribed_state",
        ocn_mode="exchange",
        ocean_t_c=(5,) * 12,
        exchange_m3s=(2.0e6,) * 12,
    ),
    "wet_westerly": dict(
        t_air_c=(5, 5, 7, 9, 12, 15, 17, 17, 15, 12, 8, 6),
        rh=(0.85,) * 12,
        atm_mode="prescribed_state",
    ),
    "dry_continental": dict(
        t_air_c=(-10, -8, -2, 7, 14, 19, 22, 20, 14, 6, -2, -8),
        rh=(0.45,) * 12,
        atm_mode="prescribed_state",
    ),
    "mountain_barrier": dict(atm_mode="closed"),
    "interior": dict(atm_mode="interior"),
}
PRESET_TITLES = {
    "interior": "как внутри (нулевой градиент, синтетика)",
    "warm_ocean": "открытый тёплый океан",
    "cold_ocean": "холодный океан",
    "wet_westerly": "влажный западный поток",
    "dry_continental": "сухой континентальный",
    "mountain_barrier": "горный барьер (закрыто)",
}


def from_preset(boundary_id: str, sides, preset: str) -> BoundaryProfile:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}")
    return BoundaryProfile(
        boundary_id, tuple(sides), preset=preset, source="preset", **PRESETS[preset]
    )


def validate(profiles) -> list[str]:
    """Errors of a profile set (empty = valid): units, ranges, double forms, coverage."""
    errs = []
    seen = {}
    for p in profiles:
        tag = f"[{p.boundary_id}]"
        for s in p.sides:
            if s not in SIDES:
                errs.append(f"{tag} unknown side {s!r}")
            if s in seen:
                errs.append(f"{tag} side {s} already covered by {seen[s]}")
            seen[s] = p.boundary_id
        if p.atm_mode not in ATM_MODES:
            errs.append(f"{tag} atm_mode must be one of {ATM_MODES}")
        if p.ocn_mode not in OCN_MODES:
            errs.append(f"{tag} ocn_mode must be one of {OCN_MODES}")
        if p.atm_mode == "prescribed_state":
            if len(p.t_air_c) != 12:
                errs.append(f"{tag} t_air_c needs 12 monthly values")
            if bool(p.q_kgkg) == bool(p.rh):
                errs.append(f"{tag} give either q_kgkg or rh (12 values), not both / neither")
            if p.moisture_flux_kg_m_s:
                errs.append(f"{tag} prescribed_state and a direct flux are mutually exclusive")
            for t in p.t_air_c:
                if not -90.0 <= t <= 60.0:
                    errs.append(f"{tag} air temperature {t} degC out of range")
            for q in p.q_kgkg:
                if not 0.0 <= q <= 0.05:
                    errs.append(f"{tag} q {q} kg/kg out of range (not g/kg?)")
            try:
                p.q_monthly()
            except ValueError as exc:
                errs.append(f"{tag} {exc}")
            if not 30000.0 <= p.p_pa <= 110000.0:
                errs.append(f"{tag} pressure {p.p_pa} Pa out of range")
        if p.atm_mode == "prescribed_flux":
            if len(p.moisture_flux_kg_m_s) != 12:
                errs.append(f"{tag} moisture_flux_kg_m_s needs 12 monthly values")
            if p.q_kgkg or p.rh:
                errs.append(f"{tag} prescribed_flux and a prescribed state are mutually exclusive")
        if p.ocn_mode == "exchange":
            if len(p.exchange_m3s) != 12 or len(p.ocean_t_c) != 12:
                errs.append(f"{tag} ocean exchange needs 12 exchange_m3s and 12 ocean_t_c")
            if any(x < 0 for x in p.exchange_m3s):
                errs.append(
                    f"{tag} exchange_m3s is a two-way exchange (>= 0); a net "
                    "inflow into a closed basin is incompatible (T17)"
                )
            if any(not -2.5 <= t <= 40.0 for t in p.ocean_t_c):
                errs.append(f"{tag} ocean temperature out of range")
    return errs


def to_json(profiles) -> str:
    """Canonical JSON (sorted keys, 6 significant digits): the same text everywhere."""

    def norm(v):
        if isinstance(v, float):
            return float(f"{v:.6g}")
        if isinstance(v, (list, tuple)):
            return [norm(x) for x in v]
        if isinstance(v, dict):
            return {k: norm(x) for k, x in sorted(v.items())}
        return v

    data = {
        "schema": "hexworld-boundary-1",
        "profiles": [norm(asdict(p)) for p in sorted(profiles, key=lambda p: p.boundary_id)],
    }
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def from_json(text: str) -> list[BoundaryProfile]:
    if not text:
        return []
    data = json.loads(text)
    if data.get("schema") != "hexworld-boundary-1":
        raise ValueError("not a hexworld-boundary-1 document")
    out = []
    for d in data.get("profiles", []):
        d = dict(d)
        for k in (
            "sides",
            "t_air_c",
            "q_kgkg",
            "rh",
            "moisture_flux_kg_m_s",
            "exchange_m3s",
            "ocean_t_c",
        ):
            d[k] = tuple(d.get(k, ()))
        out.append(BoundaryProfile(**d))
    return out


def profile_hash(text: str) -> str:
    return hashlib.sha256(to_json(from_json(text)).encode()).hexdigest()[:16] if text else ""


def side_of_raster_cells(ny: int, nx: int):
    """Raster boundary cell indices per side (rows grow southward)."""
    import numpy as np

    idx = np.arange(ny * nx).reshape(ny, nx)
    return {"west": idx[:, 0], "east": idx[:, -1], "north": idx[0], "south": idx[-1]}


def atmosphere_bc(profiles, mesh, mode: str = "isolated_approximation"):
    """Face / cell arrays of the moisture solver from the profiles.

    Returns ``(x_bc (12, n) NaN = interior donor, closed_faces (k,), bc_source (12, n)
    kg m^-2 s^-1 inward, provenance)``. ``closed_test_basin`` closes every face.
    """
    import numpy as np

    from .atmosphere.fv import boundary_sides

    n = mesh.n
    k = mesh.bi.size
    side_of = boundary_sides(mesh)
    names = np.array(SIDES)[side_of]
    x_bc = np.full((12, n), np.nan)
    closed = np.zeros(k, dtype=bool)
    src = np.zeros((12, n))
    prov = {}
    if mode == "closed_test_basin":
        return None, np.ones(k, dtype=bool), None, {"all": "closed_test_basin"}
    for p in profiles:
        sel = np.isin(names, p.sides)
        cells = mesh.bi[sel]
        if p.atm_mode == "closed":
            closed |= sel
        elif p.atm_mode == "prescribed_state":
            col = np.asarray(p.column_kg_m2())
            for m in range(12):
                x_bc[m, cells] = col[m]
        elif p.atm_mode == "prescribed_flux":
            closed |= sel
            per_area = mesh.blength[sel] / mesh.area[cells]
            for m in range(12):
                np.add.at(src[m], cells, float(p.moisture_flux_kg_m_s[m]) * per_area)
        for s in p.sides:
            prov[s] = f"{p.atm_mode}:{p.boundary_id}:{p.preset or p.source}"
    return x_bc, closed, src, prov


def ocean_exchange(atm) -> list:
    """``[(side, exchange_m3s[12], t_c[12]), ...]`` of the open ocean sides, or []."""
    text = getattr(atm, "boundary_profiles_json", "") if atm is not None else ""
    if not text or not getattr(atm, "enabled", True):
        return []
    out = []
    for p in from_json(text):
        if p.ocn_mode == "exchange":
            out.extend((s, tuple(p.exchange_m3s), tuple(p.ocean_t_c)) for s in p.sides)
    return out
