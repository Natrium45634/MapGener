"""P1 dynamics of TZ 5.5 §12–13 (G6): actual states year by year on top of the baseline.

The climate normals never change; a year's weather changes the actual state only::

    NPP_y      = NPP_b · (1 + s (P_y/P_b − 1)),  s = 1 where water limits, else 0.3
    X'         = X*_y + (X − X*_y) · exp(−dt / τ)        recovery towards this year's state
    wood       regrows with τ = 40 yr after a loss: no instant mature forest (B69)
    fire       loss = B · affected · mortality; an event id is applied once (N20, B68)
    detritus   D' = D e^{−k dt} + I (1 − e^{−k dt}) / k  (N12)
    population N' = K N / (N + (K − N) e^{−r dt});  K = 0: N e^{−m dt}   (B45, B46)
    migration  donor → receivers with shared capacity; Σ conserved except mortality (N09)

Every kgC moves by a ledger entry (donor, receiver, interval, event id); the residual
``|new − old − in + out| / max(|old|, |in| + |out|, floor)`` is reported per year.
Feedback to other layers is a typed proposal applied by the owner through a transactional
coupler; in the default one-way mode it is recorded, not applied (C09–C16).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

#: time scales of recovery of the plant pools (yr) [К]
TAU = {"leaf": 1.0, "root": 2.0, "repro": 1.0, "wood": 40.0}
#: detritus decay at 10 °C (1/yr) and Q10 [S]
K_DET_10 = 0.2
Q10 = 2.0
#: fire: mortality of the burnt share by pool [К]
FIRE_MORTALITY = {"leaf": 0.9, "repro": 0.9, "root": 0.2, "wood": 0.3}
DROUGHT_P_RATIO = 0.75
SCALE_FLOOR_KGC = 1.0


def _rng(seed: str, *key) -> np.random.Generator:
    h = hashlib.sha256("|".join(("bio_events_v1", seed, *map(str, key))).encode("ascii",
                       "backslashreplace")).digest()  # fmt: skip
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def hex_uniform(seed: str, year: int, n: int, tag: str) -> np.ndarray:
    """Uniforms per hex and year independent of traversal order (one stream per tag/year)."""
    return _rng(seed, tag, year).random(n)


# -- elementary operators (also the numerical references) -----------------------------------
def relax(x, x_star, dt, tau):
    return x_star + (np.asarray(x, dtype=np.float64) - x_star) * math.exp(-dt / tau)


def detritus_step(d, k, dt, inflow=0.0):
    """Exact first-order decay with a constant inflow (N12); returns (new, decomposed)."""
    d = np.asarray(d, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    e = np.exp(-k * dt)
    with np.errstate(invalid="ignore", divide="ignore"):
        gain = np.where(k > 0, inflow * (1.0 - e) / np.where(k > 0, k, 1.0), inflow * dt)
    new = d * e + gain
    return new, d + inflow * dt - new


def logistic_step(N, K, r, dt, m_starve=1.0):
    """Closed-form logistic toward K; K = 0 declines exponentially (no divide by zero)."""
    N = np.maximum(np.asarray(N, dtype=np.float64), 0.0)
    K = np.maximum(np.asarray(K, dtype=np.float64), 0.0)
    e = math.exp(-r * dt)
    with np.errstate(invalid="ignore", divide="ignore"):
        grow = np.where(N > 0, K * N / (N + (K - N) * e), 0.0)
    return np.where(K > 0, grow, N * math.exp(-m_starve * dt))


def migrate(donor, out_fraction, weights, capacity=None, mortality=0.0):
    """Move ``out_fraction`` of a donor stock to receivers by ``weights``; receivers share
    ``capacity`` (what does not fit stays with the donor); ``mortality`` in transit is an
    explicit loss. Returns (donor_left, received, lost) with donor0 = left + Σreceived + lost."""
    donor = float(donor)
    out = donor * float(out_fraction)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum() if w.sum() > 0 else np.zeros_like(w)
    lost = out * mortality
    arrive = (out - lost) * w
    if capacity is not None:
        cap = np.asarray(capacity, dtype=np.float64)
        fit = np.minimum(arrive, cap)
        back = float((arrive - fit).sum())
        arrive = fit
        # the surplus returns home (it never vanishes)
        return donor - out + back, arrive, lost
    return donor - out, arrive, lost


def disturbance_loss(stock, affected, mortality):
    """Loss of one disturbance on a stock: B · affected · mortality, bounded (N20)."""
    a = np.clip(np.asarray(affected, dtype=np.float64), 0.0, 1.0)
    m = np.clip(float(mortality), 0.0, 1.0)
    return np.asarray(stock, dtype=np.float64) * a * m


# -- ledger and coupler ------------------------------------------------------------------------
@dataclass
class EcoLedger:
    """Typed transfers: (transfer_id, source_pool, receiver_pool, unit, quantity, interval,
    event_id, scope); a repeated id is ignored."""

    entries: list = field(default_factory=list)
    ids: set = field(default_factory=set)

    def transfer(self, tid, src, dst, qty, unit="kgC", interval="", event_id="", scope="world"):
        if tid in self.ids:
            return False
        self.ids.add(tid)
        self.entries.append({"transfer_id": tid, "source_pool": src, "receiver_pool": dst,
                             "unit": unit, "quantity": float(qty), "interval": interval,
                             "event_id": event_id, "scope": scope})  # fmt: skip
        return True

    def totals(self) -> dict:
        out: dict = {}
        for e in self.entries:
            out[e["source_pool"]] = out.get(e["source_pool"], 0.0) - e["quantity"]
            out[e["receiver_pool"]] = out.get(e["receiver_pool"], 0.0) + e["quantity"]
        return out


def residual(old, new, inflow, outflow, floor=SCALE_FLOOR_KGC):
    num = abs(new - old - inflow + outflow)
    return num / max(abs(old), abs(inflow) + abs(outflow), floor)


@dataclass
class EcoFeedback:
    """A typed proposal to another layer's owner (never a direct write)."""

    owner: str
    field_name: str
    delta: object
    unit: str
    interval: str
    kind: str = "mass"  # mass | influence_score
    cap: float | None = None
    status: str = "proposed"
    provenance: str = "biosphere 5.5"


class Coupler:
    """Transactional application of feedback proposals: propose → validate → stage →
    commit, or roll back every owner on any failure (B77)."""

    def __init__(self, owners: dict):
        self.owners = owners  # name -> dict of numpy arrays (the owner's state)
        self.staged: dict = {}
        self.applied: set = set()

    def apply(self, proposals: list, fail_on=None) -> bool:
        snapshot = {o: {k: np.array(v, copy=True) for k, v in st.items()}
                    for o, st in self.owners.items()}  # fmt: skip
        try:
            for p in proposals:
                pid = (p.owner, p.field_name, p.interval)
                if pid in self.applied:
                    p.status = "duplicate_ignored"
                    continue
                if p.owner not in self.owners or p.field_name not in self.owners[p.owner]:
                    raise KeyError(f"unknown owner field {p.owner}.{p.field_name}")
                if fail_on is not None and fail_on(p):
                    raise RuntimeError("injected failure")
                cur = self.owners[p.owner][p.field_name]
                new = OWNER_RULES.get((p.owner, p.field_name), _add_rule)(cur, p)
                if not np.all(np.isfinite(new)):
                    raise ValueError("non-finite owner state")
                self.staged[pid] = new
            for (o, f_, _i), v in self.staged.items():
                self.owners[o][f_] = v
            for p in proposals:
                if p.status == "proposed":
                    p.status = "applied"
                    self.applied.add((p.owner, p.field_name, p.interval))
            self.staged = {}
            return True
        except Exception:
            self.owners.clear()
            self.owners.update(snapshot)
            self.staged = {}
            for p in proposals:
                if p.status == "proposed":
                    p.status = "rolled_back"
            return False


def _add_rule(cur, p):
    d = np.asarray(p.delta, dtype=np.float64)
    if p.cap is not None:
        d = np.clip(d, -p.cap, p.cap)
    return np.asarray(cur, dtype=np.float64) + d


def interception_mm(p_mm, lai, capacity_mm_per_lai=0.2, rain_days=None):
    """Owner-side canopy interception of the hydrology (C09): bounded by the rain."""
    p = np.asarray(p_mm, dtype=np.float64)
    days = np.maximum(np.asarray(rain_days if rain_days is not None else p / 8.0), 0.0)
    return np.minimum(p, capacity_mm_per_lai * np.asarray(lai) * days)


def erosion_cover_factor(cover, bare_rock=None):
    """Owner-side vegetation protection of the erosion (C15): RUSLE-like C in (0, 1];
    bare rock is not protected by plants (exception)."""
    c = np.exp(-0.05 * 100.0 * np.clip(np.asarray(cover, dtype=np.float64), 0.0, 1.0) / 2.0)
    c = np.clip(c, 0.01, 1.0)
    if bare_rock is not None:
        c = np.where(bare_rock, 1.0, c)
    return c


OWNER_RULES = {
    ("soil", "soc_kg_m2"): lambda cur, p: np.maximum(
        np.asarray(cur, float) + np.asarray(p.delta, float), 0.0
    ),
    ("hydrology", "interception_mm"): lambda cur, p: np.asarray(p.delta, float),
    ("erosion", "cover_factor"): lambda cur, p: np.clip(np.asarray(p.delta, float), 0.01, 1.0),
}


# -- the annual loop ----------------------------------------------------------------------------
def rain_anomaly(
    seed: str, years: int, shape, cell_km: float, rho: float = 0.4, sigma: float = 0.25
):
    """Regional wet / dry years: AR(1) in time, smoothed over ~900 km; P_y / P_b factors."""
    z = np.zeros(shape)
    out = []
    for y in range(1, years + 1):
        e = _rng(seed, "rain", y).standard_normal(shape)
        e = ndimage.gaussian_filter(e, max(0.5, 900.0 / cell_km), mode="nearest")
        e /= max(float(e.std()), 1e-9)
        z = rho * z + math.sqrt(1 - rho * rho) * e if y > 1 else e
        out.append(np.exp(sigma * z - 0.5 * sigma**2))
    return out


def advance_years(state, project, years: int, seed: str, rain=None, fire_scale: float = 1.0):
    """Advance the actual plant stocks, detritus and herbivore populations; the climate
    normals and every other layer stay as they were (B71). Returns the yearly ledger."""
    L = state.layers
    n = state.n
    area_m2 = state.area_km2 * 1e6
    g = state.frac["land_ground"].astype(np.float64)
    bm = project.biomes.layers if project is not None and project.biomes is not None else {}
    fri = np.nan_to_num(np.asarray(bm.get("fire_return_interval_yr", np.full(state.shape, np.inf)),
                                   dtype=np.float64).ravel(), nan=np.inf)  # fmt: skip
    t_ann = L["_env"]["tw"].astype(np.float64) * 0.5 + L["_env"]["tc"].astype(np.float64) * 0.5
    water_lim = L.get("npp_limit", np.zeros(n)) == 2
    sens = np.where(water_lim, 1.0, 0.3)
    from .registry import TERRESTRIAL_LAYERS, load

    reg = load()
    npp_b = np.zeros(n)  # land plants only (aquatic producers are not in these pools)
    for gid, d in state.groups.items():
        if "npp_gc_m2_hex" in d and reg.groups[gid].traits.get("layer") in TERRESTRIAL_LAYERS:
            npp_b += np.nan_to_num(d["npp_gc_m2_hex"].astype(np.float64))
    # baseline pools (kgC per m² of hex); the yearly NPP is split over them so that the
    # inputs equal the NPP exactly: in_p = share_p · NPP_y, X*_p = in_p · τ_p
    stocks = {p: _pool_stock(state, p) for p in ("leaf", "wood", "root", "repro")}
    flux_b = {p: stocks[p] / TAU[p] for p in stocks}
    tot_b = sum(flux_b.values())
    share = {p: np.where(tot_b > 0, flux_b[p] / np.where(tot_b > 0, tot_b, 1.0),
                         1.0 if p == "leaf" else 0.0) for p in stocks}  # fmt: skip
    age = np.where(g > 0, 200.0, 0.0)
    if rain is None:
        rain = rain_anomaly(seed, years, state.shape, math.sqrt(state.area_km2 / 0.866))
    led = EcoLedger()
    yearly = []
    drought_prev = np.zeros(n, bool)
    burned_total = np.zeros(n)
    k_det = K_DET_10 * Q10 ** ((t_ann - 10.0) / 10.0)
    k_det = np.where(t_ann < -5, 0.01, k_det)
    # litter pool in steady state with the baseline input (spin-up in closed form)
    det = np.where(g > 0, npp_b / 1000.0 / k_det, 0.0)

    def total_kgc():
        return sum(float((v * area_m2).sum()) for v in stocks.values()) + float(
            (det * area_m2).sum()
        )

    for y in range(1, years + 1):
        pf = np.asarray(rain[y - 1]).ravel()
        npp_y = np.maximum(npp_b * (1.0 + sens * (pf - 1.0)), 0.0) / 1000.0  # kgC/m²/yr
        drought = pf < DROUGHT_P_RATIO
        old_total = total_kgc()
        litter = np.zeros(n)
        for p in stocks:  # dX/dt = in − X/τ, exact over one year with a constant input
            inp_p = share[p] * npp_y
            new = relax(stocks[p], inp_p * TAU[p], 1.0, TAU[p])
            litter += inp_p * 1.0 - (new - stocks[p])
            stocks[p] = new
        # prolonged drought (two dry years running): extra mortality of wood (B26, B71)
        dd = drought & drought_prev
        kill = stocks["wood"] * 0.05 * dd
        stocks["wood"] = stocks["wood"] - kill
        litter += kill
        drought_prev = drought
        # fire: probability 1 / FRI, doubled in a dry year; one event id per hex-year (B68)
        u = hex_uniform(seed, y, n, "fire")
        lam = fire_scale / np.maximum(fri, 1.0) * np.where(drought, 2.0, 1.0)
        fire = (u < 1.0 - np.exp(-lam)) & (g > 0) & (stocks["leaf"] > 0)
        affected = np.where(fire, 0.3 + 0.4 * hex_uniform(seed, y, n, "fire_area"), 0.0)
        emitted = np.zeros(n)
        for p, mort in FIRE_MORTALITY.items():
            loss = disturbance_loss(stocks[p], affected, mort)
            stocks[p] = stocks[p] - loss
            burn = 0.8 if p in ("leaf", "repro") else 0.4
            emitted += loss * burn
            litter += loss * (1.0 - burn)
        age = np.where(g > 0, (age + 1.0) * (1.0 - affected), 0.0)
        burned_total += affected
        det, decomposed = detritus_step(det, k_det, 1.0, inflow=litter)
        new_total = total_kgc()
        npp_in = float((npp_y * area_m2).sum())
        out_fire = float((emitted * area_m2).sum())
        out_decomp = float((decomposed * area_m2).sum())
        led.transfer(f"npp/{y}", "atmosphere_prescribed", "plants", npp_in, interval=str(y))
        led.transfer(f"litter/{y}", "plants", "detritus", float((litter * area_m2).sum()),
                     interval=str(y))  # fmt: skip
        led.transfer(f"fire/{y}", "plants", "atmosphere_prescribed", out_fire,
                     interval=str(y), event_id=f"fire_year_{y}")  # fmt: skip
        led.transfer(f"decomp/{y}", "detritus", "atmosphere_prescribed", out_decomp,
                     interval=str(y))  # fmt: skip
        res = residual(old_total, new_total, npp_in, out_fire + out_decomp)
        yearly.append({"year": y, "npp_input_kgc": npp_in, "fire_emitted_kgc": out_fire,
                       "decomposed_kgc": out_decomp, "fires": int(fire.sum()),
                       "drought_hexes": int((drought & (g > 0)).sum()),
                       "stock_kgc": new_total, "residual": res})  # fmt: skip
    total = sum(stocks[p] for p in stocks)
    L["dyn_plant_biomass_kgc_m2"] = total.astype(np.float32)
    L["dyn_wood_kgc_m2"] = stocks["wood"].astype(np.float32)
    L["dyn_detritus_kgc_m2"] = det.astype(np.float32)
    L["dyn_stand_age_yr"] = age.astype(np.float32)
    L["dyn_burned_fraction"] = np.clip(burned_total, 0, None).astype(np.float32)
    L["dyn_last_year_rain_factor"] = np.asarray(rain[years - 1]).ravel().astype(np.float32)
    state.ledger["dynamic"] = {"years": yearly, "transfers": len(led.entries),
                               "max_residual": max((y_["residual"] for y_ in yearly),
                                                   default=0.0)}  # fmt: skip
    state.summary["dynamic_years"] = years
    # typed feedback proposals (one-way: recorded, not applied)
    lai = np.zeros(n)
    for gid, d in state.groups.items():
        if "cover" in d:
            from .registry import load

            tr = load().groups[gid].traits
            if tr.get("layer") in ("canopy", "shrub", "ground", "epiphyte"):
                lai += d["cover"].astype(np.float64) * tr["lai_max"] * g
    state.ledger["feedback"] = [
        {"owner": "hydrology", "field": "interception_mm", "unit": "mm/yr",
         "status": "proposed_one_way", "mean": float(interception_mm(800.0, lai).mean())},
        {"owner": "soil", "field": "soc_kg_m2", "unit": "kgC/m²",
         "status": "proposed_one_way", "mean": float(det.mean())},
        {"owner": "erosion", "field": "cover_factor", "unit": "1",
         "status": "proposed_one_way",
         "mean": float(erosion_cover_factor(np.clip(lai / 4.0, 0, 1)).mean())},
    ]  # fmt: skip
    return yearly


def _pool_stock(state, pool: str) -> np.ndarray:
    """Baseline stock of one plant pool (kgC per m² of hex) from the stored layers."""
    if pool == "wood":
        return state.layers["wood_kgc_m2"].astype(np.float64).copy()
    total = state.layers["plant_biomass_kgc_m2"].astype(np.float64)
    wood = state.layers["wood_kgc_m2"].astype(np.float64)
    rest = np.maximum(total - wood, 0.0)
    share = {"leaf": 0.3, "root": 0.6, "repro": 0.1}[pool]
    return rest * share
