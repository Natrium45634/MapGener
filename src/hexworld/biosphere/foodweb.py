"""Joint food allocation and consumer capacities of TZ 5.5 §10 (G4).

Every pool r (kgC/yr per hex) is shared by all its consumers at once::

    D_g      = w_g · d0_g · A_home                         one total demand per group (B42)
    req_g,r  = min(D_g · diet_g,r, φ_g · A_r)
    λ_r      = min(1, A_r / Σ_g req_g,r)                   (N10, B40)
    I_g      = Σ_r λ_r req_g,r
    assimilated = a_g I_g,  production P_g = npe_g · a_g · I_g  (≤ assimilated, B44)
    K        = min(K_food, K_space, K_water)               (individuals)
    B_g      = P_g / (P/B)_g,  N_g = B_g / (M_g · c_wet)

Pools fed by animals become available only after all their providers are done, and only to
consumers of a higher level (checked by the registry). Uneaten production goes to the
detritus or carrion once (B47). The ledger closes per hex: every kgC in a pool ends in
exactly one of allocated / left over, every intake in egested + respired + produced.
"""

from __future__ import annotations

import numpy as np

from .registry import load, pool_provider_level

#: carbon per wet mass of an animal [S] and energy per kg of carbon (MJ/kgC) [S]
C_PER_WET = 0.12
MJ_PER_KGC = 42.0
#: share of the uneaten production of land vertebrates (≥ 0.5 kg) left as carrion [К]
CARRION_SHARE = 0.3
#: upper density by body mass (Damuth-like), individuals/km² = DMAX · M^-0.75 [К]
DMAX = 200.0
VERTEBRATE_KG = 0.005
#: aquatic animals are not capped by a land territory rule
WATER_REALMS = ("freshwater", "marine", "marine_freshwater", "aquatic")


def allocate(available, requests):
    """Proportional cap of the requests to one pool: returns the allocations and λ (N10)."""
    a = np.asarray(available, dtype=np.float64)
    req = [np.asarray(r, dtype=np.float64) for r in requests]
    tot = np.sum(req, axis=0) if req else np.zeros_like(a)
    with np.errstate(invalid="ignore", divide="ignore"):
        lam = np.where(tot > a, a / np.where(tot > 0, tot, 1.0), 1.0)
    lam = np.clip(np.nan_to_num(lam, nan=0.0), 0.0, 1.0)
    return [lam * r for r in req], lam


def energy_split(intake_kgc, assim, npe):
    """kgC intake → (assimilated, egested, respired, production) kgC and the energies (B44)."""
    i = np.asarray(intake_kgc, dtype=np.float64)
    if not 0 < assim <= 1 or not 0 <= npe <= 1:
        raise ValueError("assimilation and production efficiency must be in (0, 1]")
    a = assim * i
    prod = npe * a
    return {"assimilated": a, "egested": i - a, "respired": a - prod, "production": prod,
            "assimilated_mj": a * MJ_PER_KGC}  # fmt: skip


def run_foodweb(pools: dict, weights: dict, home_area_m2: dict, n: int, gates_water=None,
                home_idx: dict | None = None):  # fmt: skip
    """Joint allocation over the hexes.

    ``pools``: primary pool -> kgC/yr per hex (N). ``weights`` and ``home_area_m2``: consumer
    group -> w in [0, 1] and habitat area per hex, either full (N) or compact on
    ``home_idx[group]`` (the hexes of its realm). Results are compact on the same hexes
    (float32): memory grows with the realm, not with every hex of the world (§16)."""
    reg = load()
    cons = [g for g in reg.groups.values() if not g.is_producer]
    full = np.arange(n)
    hidx = {g.id: (home_idx[g.id] if home_idx is not None else full) for g in cons}
    A = {
        k: np.asarray(v, dtype=np.float64).copy() for k, v in pools.items() if not k.startswith("_")
    }
    for p in reg.pools:
        A.setdefault(p, np.zeros(n))
    pool_in = {k: float(v.sum()) for k, v in A.items()}
    eaten_share: dict = {}  # pool -> share of it allocated (for the carrion of the uneaten)
    intake_acc: dict = {}  # allocations summed per consumer, compact on its hexes
    res = {g.id: {} for g in cons}
    led = {"pool_in": 0.0, "allocated": 0.0, "left_over": 0.0, "intake": 0.0, "egested": 0.0,
           "respired": 0.0, "production": 0.0, "returned_unused": 0.0,
           "carrion_from_uneaten": 0.0}  # fmt: skip

    def demand(c):
        tr = reg.groups[c].traits
        w = np.clip(np.nan_to_num(np.asarray(weights.get(c, 0.0), dtype=np.float64)), 0.0, 1.0)
        return w * tr["d0"] * np.asarray(home_area_m2[c], dtype=np.float64) / 1000.0  # kgC/yr

    levels = sorted({g.level for g in cons})
    done_pools: set = set()

    def allocate_ready(level):
        for p in sorted(reg.pools):
            if p in done_pools:
                continue
            consumers = [c for c in reg.pools[p]["consumers"]
                         if reg.groups[c].traits["diet"].get(p, 0) > 0]  # fmt: skip
            if not consumers:
                done_pools.add(p)
                continue
            if pool_provider_level(reg, p) >= level:
                continue
            if p == "carrion":
                _carrion()
            avail = A[p]
            tot = np.zeros(n)
            reqs = []
            for c in consumers:
                tr = reg.groups[c].traits
                ix = hidx[c]
                r = np.minimum(demand(c) * tr["diet"][p], tr["phi"] * avail[ix])
                reqs.append(r)
                np.add.at(tot, ix, r) if ix is not full else np.add(tot, r, out=tot)
            with np.errstate(invalid="ignore", divide="ignore"):
                lam = np.where(tot > avail, avail / np.where(tot > 0, tot, 1.0), 1.0)
            lam = np.clip(np.nan_to_num(lam, nan=0.0), 0.0, 1.0)
            used = np.zeros(n)
            for c, r in zip(consumers, reqs, strict=True):
                ix = hidx[c]
                a = lam[ix] * r
                intake_acc[c] = intake_acc[c] + a if c in intake_acc else a
                np.add.at(used, ix, a) if ix is not full else np.add(used, a, out=used)
            with np.errstate(invalid="ignore", divide="ignore"):
                eaten_share[p] = np.where(avail > 0, used / np.where(avail > 0, avail, 1.0),
                                          0.0).astype(np.float32)  # fmt: skip
            led["allocated"] += float(used.sum())
            led["left_over"] += float((avail - used).sum())
            done_pools.add(p)

    def _carrion():
        tot = np.zeros(n)
        for g in cons:
            tr = g.traits
            if g.realm not in ("land", "freshwater_land") or tr["mass_kg"] < 0.5:
                continue
            prod = res[g.id].get("production")
            if prod is None:
                continue
            p = tr.get("provides")
            ix = hidx[g.id]
            share = eaten_share[p][ix] if p in eaten_share else 0.0
            uneaten = prod.astype(np.float64) * (1.0 - np.clip(share, 0.0, 1.0))
            np.add.at(tot, ix, CARRION_SHARE * uneaten)
        A["carrion"] = A["carrion"] + tot
        pool_in["carrion"] = float(A["carrion"].sum())
        led["carrion_from_uneaten"] += float(tot.sum())

    for level in levels:
        allocate_ready(level)
        for g in [c for c in cons if c.level == level]:
            tr = g.traits
            ix = hidx[g.id]
            intake = intake_acc.pop(g.id, np.zeros(ix.size))
            es = energy_split(intake, tr["assim"], tr["npe"])
            biomass = es["production"] / tr["pb"]  # kgC
            m = tr["mass_kg"]
            area_km2 = np.asarray(home_area_m2[g.id], dtype=np.float64) / 1e6
            n_food = biomass / max(m * C_PER_WET, 1e-30)
            # space cap (Damuth-like) only for land vertebrates; invertebrates, plankton,
            # microbes and aquatic animals are food-limited here
            n_space = DMAX * max(m, 1e-9) ** -0.75 * area_km2 \
                if m >= VERTEBRATE_KG and g.realm not in WATER_REALMS else \
                np.full(ix.size, np.inf)  # fmt: skip
            k = np.minimum(n_food, n_space)
            if tr.get("needs_water") and gates_water is not None:
                k = np.where(gates_water[ix], k, 0.0)  # no drinking water: K_water = 0
            scale = np.where(n_food > 0, k / np.maximum(n_food, 1e-30), 0.0)
            used = intake * scale
            unused = intake - used
            es = energy_split(used, tr["assim"], tr["npe"])
            res[g.id] = {
                "intake": used.astype(np.float32),
                "production": es["production"].astype(np.float32),
                "biomass_kgc": (es["production"] / tr["pb"]).astype(np.float32),
                "individuals": np.asarray(k, dtype=np.float32),
                "limit": (np.where(n_food <= n_space, 1, 2) * (k > 0)).astype(np.uint8),
            }
            led["intake"] += float(used.sum())
            led["egested"] += float(es["egested"].sum())
            led["respired"] += float(es["respired"].sum())
            led["production"] += float(es["production"].sum())
            led["returned_unused"] += float(unused.sum())
            if tr.get("provides"):
                pv = tr["provides"]
                np.add.at(A[pv], ix, es["production"])
                pool_in[pv] = float(A[pv].sum())
    allocate_ready(max(levels) + 1)
    led["pool_in"] = float(sum(pool_in.values()))
    for p, v in pool_in.items():
        if p not in eaten_share:  # nobody eats it: all of it stays (detritus, export)
            led["left_over"] += v
    # closure: allocated = intake + returned unused; intake = egested + respired + production
    led["residual_allocation"] = led["allocated"] - led["intake"] - led["returned_unused"]
    led["residual_energy"] = led["intake"] - led["egested"] - led["respired"] - led["production"]
    scale = max(led["pool_in"], 1.0)
    led["ok"] = (abs(led["residual_allocation"]) / scale < 1e-8
                 and abs(led["residual_energy"]) / scale < 1e-8)  # fmt: skip
    return {"groups": res, "eaten_share": eaten_share, "ledger": led, "home_idx": hidx}
