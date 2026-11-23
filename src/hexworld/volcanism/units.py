"""Units, constants and helpers of the volcanism package (SI unless the name says otherwise).

The constants are a starting earthlike calibration ``[H]``, not universal properties.
"""

from __future__ import annotations

import hashlib
import math

SECONDS_PER_YEAR = 365.25 * 86400.0
G = 9.81  # m/s^2
RHO_SEAWATER = 1025.0  # kg/m^3
P_ATM = 101_325.0  # Pa
L_FUSION_ICE = 334_000.0  # J/kg
RHO_ICE = 917.0  # kg/m^3
RHO_WATER = 1000.0  # kg/m^3
#: bulk (loose) density of fallen tephra, kg/m^3 - used only after deposition
RHO_TEPHRA_BULK = 1000.0
#: dense-rock-equivalent density of magma by default, kg/m^3
RHO_MAGMA_DRE = 2600.0
#: SO2 is twice the mass of the sulphur it carries
SO2_PER_S = 64.07 / 32.07
MOLAR_SO2 = 0.06407  # kg/mol

#: Classic VEI thresholds of bulk tephra volume, m^3 (VEI 2 ... 8); a reference only.
VEI_BULK_M3 = (1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12)


def stable_hash(*parts) -> str:
    """Hex digest of the parts (deterministic across runs and platforms)."""
    text = "|".join(str(p) for p in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_uniform(*parts) -> float:
    """A uniform number in (0, 1) fixed by the parts (no hidden RNG state)."""
    h = int(stable_hash(*parts)[:15], 16)
    return (h + 0.5) / float(16**15)


def seawater_pressure_pa(depth_m: float) -> float:
    """Ambient pressure at a vent ``depth_m`` below the sea surface, Pa."""
    return P_ATM + RHO_SEAWATER * G * max(0.0, float(depth_m))


def vei_from_bulk(bulk_m3: float) -> int:
    """VEI 0..8 from the bulk tephra volume (0 below 1e4 m^3 / effusive)."""
    if bulk_m3 < 1e4:
        return 0
    if bulk_m3 < 1e6:
        return 1
    v = 2
    for thr in VEI_BULK_M3[1:]:
        if bulk_m3 >= thr:
            v += 1
    return min(v, 8)


def plume_height_m(dre_rate_m3_s: float) -> float:
    """Eruption column top from the DRE eruption rate (Mastin et al. 2009, H = 2.00 Q^0.241 km)."""
    if dre_rate_m3_s <= 0:
        return 0.0
    return 1000.0 * 2.0 * dre_rate_m3_s**0.241


def tropopause_m(lat_deg: float) -> float:
    """Approximate tropopause height, m: ~16 km in the tropics, ~9 km near the poles."""
    phi = abs(float(lat_deg))
    return 9000.0 + 7000.0 * math.exp(-((phi / 35.0) ** 2))
