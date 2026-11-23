"""Floating ice shelves: grounding line, flotation, basal melt, calving (TZ 5.3 sec. 9, P0).

A shelf is the **floating continuation of grounded ice** - never frozen sea water. Only
ocean hexes that receive ice flux from a grounded sheet can hold a shelf (T42).

Flotation (hydrostatic): ice of thickness ``H`` floats in water of depth ``D`` when
``rho_i H <= rho_sw D``; draft ``= rho_i/rho_sw H``, freeboard ``= H (1 - rho_i/rho_sw)``
(500 m of ice: 446 m draft, 54 m freeboard).

P0 flux-constrained shelf (``[ADAPT]``; buttressing and SSA flow are not modelled): the flux
``F`` crossing the grounding line (m^3/yr, from the SIA solver) enters the first shelf hex;
going seaward hex by hex (breadth-first in distance from the grounding line) each shelf hex
loses ice by spreading-thinning ``H_next = H exp(-ds / L_spread)``, basal melt
``m = m0 + k max(T_ocean - T_f, 0)`` (m/yr; the ocean heat used is reported) and passes the
rest on; where the thickness falls below ``H_front`` or the water is warm (annual SST > 0)
the remaining flux calves into the iceberg ledger. Per hex
``F_in = F_out + melt + calving`` exactly; the ledger closes to round-off.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

RHO_ICE = 917.0
RHO_SEA = 1028.0
L_FUSION = 334_000.0
T_FREEZE = -1.8
L_SPREAD_M = 150e3
H_FRONT_M = 150.0
MELT0_M_YR = 0.1
MELT_K_M_YR_K = 0.8


def draft_freeboard(h_m):
    h = np.asarray(h_m, dtype=np.float64)
    return h * RHO_ICE / RHO_SEA, h * (1.0 - RHO_ICE / RHO_SEA)


def floats(h_m, depth_m):
    return RHO_ICE * np.asarray(h_m) <= RHO_SEA * np.asarray(depth_m)


@dataclass
class ShelfResult:
    thickness_m: np.ndarray  # (n,) shelf ice on ocean hexes
    grounding_line: np.ndarray  # (n,) bool, grounded ice hexes feeding a shelf
    ledger: dict = field(default_factory=dict)


def grounding_flux(h, bed, ocean, nbr, spacing_m, a_yr=None):
    """Ice flux (m^3/yr) leaving every grounded hex into each ocean neighbour (SIA)."""
    from .icesheet import GLEN_A_YR, glen_gamma

    gam = glen_gamma(GLEN_A_YR if a_yr is None else a_yr)
    face = spacing_m / np.sqrt(3.0)
    out = {}
    s = np.where(ocean, np.maximum(h * (1 - RHO_ICE / RHO_SEA), 0.0), bed + h)
    for k in range(6):
        j = nbr[:, k]
        ok = (j >= 0) & ~ocean & (h > 1.0)
        i = np.nonzero(ok)[0]
        jj = j[i]
        sea = ocean[jj]
        i, jj = i[sea], jj[sea]
        grad = (s[jj] - s[i]) / spacing_m
        hm = h[i]
        q = gam * hm**5 * grad**2 * np.maximum(-grad, 0.0) / np.maximum(np.abs(grad), 1e-12)
        q = q * face
        for a, b, f in zip(i.tolist(), jj.tolist(), q.tolist(), strict=True):
            if f > 0:
                out[(a, b)] = out.get((a, b), 0.0) + f
    return out


def build_shelves(h_grounded, bed, depth, ocean, sst_annual, nbr, spacing_m,
                  flux_in: dict | None = None, inflow_hex=None) -> ShelfResult:  # fmt: skip
    n = h_grounded.size
    area = np.sqrt(3.0) / 2.0 * spacing_m**2
    flux = {}
    if inflow_hex is None:
        flux = flux_in if flux_in is not None else grounding_flux(h_grounded, bed, ocean, nbr,
                                                                   spacing_m)  # fmt: skip
    shelf = np.zeros(n)
    gl = np.zeros(n, dtype=bool)
    ledger = {"grounding_flux_m3_yr": 0.0, "basal_melt_m3_yr": 0.0, "calving_m3_yr": 0.0,
              "ocean_heat_used_w": 0.0, "spreading_m3_yr": 0.0}  # fmt: skip
    # inflow per ocean hex and the grounded thickness at the line
    inflow = np.zeros(n)
    h_line = np.zeros(n)
    if inflow_hex is not None:
        # the sheet solver's own discharge into each ocean hex (m^3/yr)
        inflow = np.where(ocean, np.maximum(np.asarray(inflow_hex, dtype=np.float64), 0.0), 0.0)
        for b in np.nonzero(inflow > 0)[0]:
            for k in range(6):
                a = int(nbr[b, k])
                if a >= 0 and not ocean[a] and h_grounded[a] > 1.0:
                    gl[a] = True
                    h_line[b] = max(h_line[b], h_grounded[a])
    else:
        for (a, b), f in flux.items():
            inflow[b] += f
            h_line[b] = max(h_line[b], h_grounded[a])
            gl[a] = True
    ledger["grounding_flux_m3_yr"] = float(inflow.sum())
    if inflow.sum() <= 0:
        return ShelfResult(shelf, gl, ledger)
    # breadth-first seaward: each shelf hex passes its remaining flux to its ocean
    # neighbours that are farther from the grounding line
    dist = np.full(n, -1, dtype=np.int64)
    q = deque()
    for b in np.nonzero(inflow > 0)[0]:
        dist[b] = 0
        q.append(int(b))
    order = []
    while q:
        a = q.popleft()
        order.append(a)
        for k in range(6):
            b = int(nbr[a, k])
            if b >= 0 and ocean[b] and dist[b] < 0:
                dist[b] = dist[a] + 1
                q.append(b)
    flow = inflow.copy()
    thick = np.where(inflow > 0, h_line * 0.9, 0.0)
    sst = np.nan_to_num(np.asarray(sst_annual, dtype=np.float64), nan=T_FREEZE)
    for a in order:
        f_in = flow[a]
        if f_in <= 0:
            continue
        h_here = thick[a]
        warm = sst[a] > 0.0
        if h_here < H_FRONT_M or warm or not floats(h_here, max(depth[a], 1.0)):
            ledger["calving_m3_yr"] += f_in  # the front: everything left becomes icebergs
            flow[a] = 0.0
            continue
        shelf[a] = h_here
        melt_rate = MELT0_M_YR + MELT_K_M_YR_K * max(sst[a] - T_FREEZE, 0.0)
        melt = min(f_in, melt_rate * area)
        ledger["basal_melt_m3_yr"] += melt
        ledger["ocean_heat_used_w"] += melt * RHO_ICE * L_FUSION / (365.25 * 86400.0)
        rest = f_in - melt
        nxt = []
        for k in range(6):
            b = int(nbr[a, k])
            if b >= 0 and ocean[b] and dist[b] == dist[a] + 1:
                nxt.append(b)
        if not nxt:
            ledger["calving_m3_yr"] += rest
            continue
        share = rest / len(nxt)
        h_next = h_here * np.exp(-spacing_m / L_SPREAD_M)
        for b in nxt:
            flow[b] += share
            thick[b] = max(thick[b], h_next)
    ledger["residual_m3_yr"] = ledger["grounding_flux_m3_yr"] - ledger["basal_melt_m3_yr"] - \
        ledger["calving_m3_yr"]  # fmt: skip
    return ShelfResult(shelf, gl, ledger)
