"""HexWorld v5.2: soil fertility and geological resources (stage ``v52``, after v5.0).

Inputs (never modified): v3 rock / crust age / orogens / erosion, v4.6 basins and sea
floor, v4 drainage, erosion rate, lakes, v4.5 soils and 12-month water, v5.3 upwelling,
v5.1 volcano systems, tephra and lava, v5.0 cover.

Steps::

    5.2.A  geological resources (geologic time)
        lithology adapter -> permissive tracts -> object-anchored + tract systems
        -> parametric bodies -> exhumation / burial -> in-situ grade and tonnage
    5.2.B  soil fertility (ecological years)
        B0 soil age and parent chemistry -> B1 weathering / CEC / pH / salts
        -> B2 C, N, P, K pools -> erosion transport of soil elements (ledger)
        -> B3 second pass with the alluvial inputs + monthly consistency check
        -> B4 suitability per vegetation / crop profile
    5.2.A' secondary: placers from eroded bodies, laterites (needs B1), coastal sands,
        closed-basin salts and Li brines, construction materials
    aggregates -> dense hex layers + sparse catalogue + ledgers

The two time scales stay separate: deposits are fixed geological events (Ma), soil
processes are years over the soil age; 40 Ma never become 40 growth cycles.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict

import numpy as np

from .project import ResourceData
from .resources import ALGORITHM_VERSION
from .resources import deposits as dep
from .resources import secondary as sec
from .resources import soil as sl
from .resources.models import FAMILIES

#: crop / vegetation profiles stored as layers (others are computed on demand)
STORED_PROFILES = (
    "native_generic",
    "forest",
    "grassland",
    "crop_wheat",
    "crop_rice_paddy",
    "crop_maize",
    "crop_cassava",
)

RESOURCE_LAYER_SPECS = {
    "soil_age_yr": ("yr", "years since the last surface reset (not the rock age)"),
    "soil_layer_bottom_m": ("m", "(3, H, W) bottoms of the top / sub / deep functional layers"),
    "soil_bulk_density_kg_m3": ("kg/m3", "(3, H, W) dry fine-earth bulk density"),
    "soil_soc_layers_kg_m2": ("kg C/m2", "(3, H, W) organic carbon stock of each layer"),
    "soil_ph_layers": ("pH", "(3, H, W) pH per layer (never averaged linearly)"),
    "soil_soc_kg_m2": ("kg C/m2", "profile organic carbon stock (RothC pools + IOM)"),
    "soil_organic_N_kg_m2": ("kg N/m2", "organic nitrogen"),
    "soil_ammonium_N_kg_m2": ("kg N/m2", "NH4-N (steady state)"),
    "soil_nitrate_N_kg_m2": ("kg N/m2", "NO3-N (steady state)"),
    "soil_N_net_mineralisation_kg_m2_yr": (
        "kg N/m2/yr",
        "net mineralisation (negative = immobilisation)",
    ),
    "soil_P_primary_kg_m2": ("kg P/m2", "apatite / primary mineral P"),
    "soil_P_labile_kg_m2": ("kg P/m2", "labile (plant-available) P"),
    "soil_P_sorbed_kg_m2": ("kg P/m2", "sorbed, non-occluded P"),
    "soil_P_organic_kg_m2": ("kg P/m2", "organic P"),
    "soil_P_occluded_kg_m2": ("kg P/m2", "occluded by Fe / Al oxides"),
    "soil_K_mineral_kg_m2": ("kg K/m2", "K in primary minerals"),
    "soil_K_exchange_kg_m2": ("kg K/m2", "exchangeable K"),
    "soil_K_solution_kg_m2": ("kg K/m2", "K in the soil solution"),
    "soil_cec_cmolc_kg": ("cmol(+)/kg", "cation exchange capacity at pH 7"),
    "soil_base_saturation_frac": ("0..1", "base saturation of the exchange complex"),
    "soil_ph": ("pH", "profile pH (buffer-curve proxy)"),
    "soil_acid_equivalent_molc_m2": ("molc/m2", "exchangeable acidity"),
    "soil_buffer_capacity_molc_m2_per_ph": ("molc/m2/pH", "acid buffering"),
    "soil_carbonate_kg_m2": ("kg CaCO3/m2", "carbonate left in the profile"),
    "soil_salt_mass_kg_m2": ("kg/m2", "soluble salt mass (not an index)"),
    "soil_salinity_ec_ds_m": ("dS/m", "EC of the saturation extract from the salt mass (proxy)"),
    "soil_esp_pct": ("%", "exchangeable sodium (sodicity proxy)"),
    "soil_oxic_fraction": ("0..1", "share of the growing time with oxygen in the root zone"),
    "soil_weathering_degree": ("0..1", "share of weatherable minerals gone"),
    "soil_erosion_risk": ("0..1", "risk of losing the fertile top layer"),
    "soil_nutrient_supply_index": ("0..1", "available N / P / K and their renewal"),
    "soil_water_supply_index": ("0..1", "AWC and how full the store is"),
    "soil_rooting_index": ("0..1", "depth, stones, soil temperature"),
    "soil_fertility_potential": ("0..1", "natural potential productivity of the soil (diagnostic)"),
    "soil_limiting_factor": (
        "enum",
        "dominant limitation of the native profile, resources.soil.FACTORS",
    ),
    "geo_lithology_inferred": ("enum", "lithology adapter, resources.deposits.LITHOLOGY"),
    "prospectivity_by_family": ("0..1", "(F, H, W) geological favourability, not proven ore"),
    "resource_in_situ_t_by_family": ("t", "(F, H, W) contained commodity in situ (no economics)"),
    "deposit_count": ("count", "deposit systems with a body under the hex"),
    "shallowest_body_depth_m": ("m", "depth of the shallowest body top below the surface"),
    "surface_occurrence_mask": ("bool", "a body crops out (visible without exploration)"),
    "overburden_m": ("m", "sediment cover thickness"),
    "resource_accessibility_proxy": ("0..1", "depth / water depth accessibility (not economics)"),
    "construction_material_class": ("enum", "resources.secondary.CONSTRUCTION"),
    "construction_material_quality": ("0..1", "relative quality of the material"),
}
for _p in STORED_PROFILES:
    RESOURCE_LAYER_SPECS[f"suitability_{_p}"] = ("0..1", f"edaphic suitability, profile {_p}")


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def resource_stage_keys(config, project) -> dict:
    up = {
        "hist": project.hist.stage_keys,
        "hydro": project.hydro.stage_keys,
        "env": project.env.stage_keys,
        "terrain": project.terrain.stage_keys if project.terrain is not None else {},
        "volcanism": project.volcanism.stage_keys if project.volcanism is not None else {},
        "biomes": project.biomes.stage_keys if project.biomes is not None else {},
        "earth": project.earth.stage_keys if project.earth is not None else {},
    }
    return {"v52_resources": _key("v52", ALGORITHM_VERSION, up, asdict(config.resources))}


def _soil_age(project, env, flat, depth, erosion_m_yr):
    """Years since the last surface reset [A]: residence time D / E of the regolith,
    capped by the last glaciation, young lava / volcanic rock, active floodplains and dunes."""
    from .stages.environment import PM

    resid = np.where(
        erosion_m_yr > 1e-9, np.maximum(depth, 0.05) / np.maximum(erosion_m_yr, 1e-12), 4e6
    )
    age = np.clip(resid, 300.0, 4e6)
    lgm = (
        flat(project.hydro.layers.get("lgm_ice_mask", np.zeros(project.layout.width * 0 + 1)), bool)
        if "lgm_ice_mask" in project.hydro.layers
        else np.zeros(age.size, bool)
    )
    age = np.where(lgm, np.minimum(age, 15000.0), age)
    v_age = np.nan_to_num(flat(project.hist.layers["volcanic_age_ma"]), nan=1e9)
    age = np.where(v_age < 2.0, np.minimum(age, np.maximum(v_age * 1e6, 300.0)), age)
    pm = flat(env["parent_material_class"], np.int64)
    fp = np.nan_to_num(flat(env["floodplain_potential"]))
    age = np.where((pm == PM["alluvium"]) & (fp > 0.4), np.minimum(age, 3000.0), age)
    age = np.where(pm == PM["aeolian_sand"], np.minimum(age, 20000.0), age)
    if project.volcanism is not None:
        V = project.volcanism.layers
        lava = np.nan_to_num(
            flat(V.get("lava_fraction", np.zeros(1)))
            if "lava_fraction" in V
            else np.zeros(age.size)
        )
        ly = (
            flat(V["lava_youngest_years_bp"])
            if "lava_youngest_years_bp" in V
            else np.full(age.size, np.nan)
        )
        young = (lava > 0.5) & np.isfinite(ly)
        age = np.where(young, np.minimum(age, np.maximum(ly, 50.0)), age)
    return age


def run_resources(config, project, report=lambda f: None) -> dict:
    prm = config.resources
    t0 = time.perf_counter()
    timings = {}
    lay = project.layout
    shape = (lay.height, lay.width)
    n = shape[0] * shape[1]

    def flat(a, dtype=np.float64):
        return (
            np.asarray(a, dtype=dtype).reshape(-1)[:n]
            if np.asarray(a).size >= n
            else np.broadcast_to(np.asarray(a, dtype=dtype).reshape(-1)[0], (n,)).copy()
        )

    env = project.env.layers
    mon = project.env.monthly
    H = project.hydro.layers
    ctx = dep.context_from_project(project)
    timings["context"] = time.perf_counter() - t0

    # -- 5.2.B soil inputs ---------------------------------------------------------------------
    t1 = time.perf_counter()
    present = flat(env["soil_present"], bool)
    depth = np.nan_to_num(flat(env["soil_depth_m"]))
    erosion_m_yr = np.maximum(np.nan_to_num(flat(H["erosion_rate_m_ma"])), 0.0) * 1e-6
    age = _soil_age(project, env, flat, depth, erosion_m_yr)
    bio = project.biomes.layers if project.biomes is not None else {}
    veg = (
        np.nan_to_num(flat(bio["vegetation_fraction"]))
        if "vegetation_fraction" in bio
        else np.clip(np.nan_to_num(flat(env["aet_over_pet"])), 0, 1)
    )
    tree = (
        np.nan_to_num(flat(bio["tree_canopy_fraction"]))
        if "tree_canopy_fraction" in bio
        else 0.3 * veg
    )
    lith = ctx.lith
    carb_extra = np.where(lith == dep.LITH["carbonate_platform"], 0.15, 0.0)
    ash = tephra = None
    if project.volcanism is not None:
        V = project.volcanism.layers
        ash = np.nan_to_num(flat(V["volcanic_ash_stress"])) if "volcanic_ash_stress" in V else None
        tephra = np.nan_to_num(flat(V["tephra_thickness_m"])) if "tephra_thickness_m" in V else None
    lake = flat(H["lake_id"], np.int64) >= 0
    endo_floor = (
        flat(H["endorheic_mask"], bool)
        & (lake | (np.nan_to_num(flat(env["aridity_p_over_pet"]), nan=1) < 0.3))
        & (flat(env["wetness_potential"]) > 0.3)
    )
    m12 = lambda k: np.asarray(mon[k], dtype=np.float64).reshape(12, -1)  # noqa: E731
    inp = sl.FertilityInputs(
        soil_present=present,
        parent=flat(env["parent_material_class"], np.int64),
        sand=flat(env["sand_frac"]),
        clay=flat(env["clay_frac"]),
        coarse=flat(env["coarse_fragment_frac"]),
        depth_m=depth,
        porosity_mineral=flat(env["porosity_vol"]),
        fc=flat(env["field_capacity_vol"]),
        pwp=flat(env["wilting_point_vol"]),
        t_month=m12("temperature_monthly_c"),
        p_month=m12("precip_monthly_mm"),
        pet_month=m12("pet_monthly_mm"),
        storage_month=m12("soil_storage_monthly_mm"),
        awc_mm=np.nan_to_num(flat(env["awc_mm"])),
        percolation_mm_yr=np.nan_to_num(flat(env["groundwater_recharge_annual_mm"])),
        waterlogging_months=np.nan_to_num(flat(env["waterlogging_months"])),
        vegetation_fraction=veg,
        tree_fraction=tree,
        soil_erosion_m_yr=erosion_m_yr,
        soil_age_yr=age,
        marine_influence=np.nan_to_num(flat(env["coastal_marine_influence"])),
        endorheic_floor=endo_floor,
        carbonate_extra=carb_extra,
        ash_stress=ash,
        tephra_m=tephra,
    )
    soil_prm = sl.SoilChemParams(alpha=prm.suitability_alpha)
    st1 = sl.compute_fertility_static(inp, soil_prm)
    timings["soil_pass1"] = time.perf_counter() - t1
    report(0.25)

    # -- erosion transport of soil elements down the rivers (ledger, S23) -------------------------
    t2 = time.perf_counter()
    area_m2 = lay.hex_area_km2 * 1e6
    fe_top = np.nan_to_num(st1["bulk_density_kg_m3"][0]) * (1.0 - np.nan_to_num(inp.coarse))
    total_fe = np.maximum(
        np.nan_to_num((st1["layer_thickness_m"] * st1["bulk_density_kg_m3"]).sum(axis=0))
        * (1 - np.nan_to_num(inp.coarse)),
        1e-9,
    )
    p_tot = np.nan_to_num(
        st1["p_primary_kg_m2"]
        + st1["p_labile_kg_m2"]
        + st1["p_sorbed_kg_m2"]
        + st1["p_organic_kg_m2"]
        + st1["p_occluded_kg_m2"]
    )
    eroded_soil_kg_m2_yr = np.where(present, erosion_m_yr * fe_top, 0.0)  # kg fine earth / m2 / yr
    src = (
        np.stack(
            [
                eroded_soil_kg_m2_yr * p_tot / total_fe,  # P
                eroded_soil_kg_m2_yr
                * np.nan_to_num(st1["soc_layers_kg_m2"][0])
                / np.maximum(np.nan_to_num(st1["layer_thickness_m"][0]) * fe_top, 1e-9),  # C
            ],
            axis=1,
        )
        * area_m2
    )  # kg/yr per hex
    src = np.nan_to_num(src)
    fp = np.clip(np.nan_to_num(flat(env["floodplain_potential"])), 0.0, 1.0)
    ocean = ctx.ocean
    retention = np.where(lake, 1.0, np.where(ocean, 0.0, 0.5 * fp))
    receiver = flat(H["flow_receiver"], np.int64)
    deposited, exported = sec.retention_route(receiver, src, retention)
    transport_ledger = {
        el: {
            "eroded_kg_yr": float(src[:, k].sum()),
            "deposited_kg_yr": float(deposited[:, k].sum()),
            "exported_kg_yr": float(exported[k]),
            "residual_kg_yr": float(src[:, k].sum() - deposited[:, k].sum() - exported[k]),
        }
        for k, el in enumerate(("P", "C"))
    }
    timings["transport"] = time.perf_counter() - t2

    # -- second pass with the alluvial inputs (B3) -----------------------------------------------
    t3 = time.perf_counter()
    inp.deposition_p_kg_m2_yr = np.where(present & ~lake, deposited[:, 0] / area_m2, 0.0)
    inp.deposition_c_kg_m2_yr = np.where(present & ~lake, deposited[:, 1] / area_m2, 0.0)
    st = sl.compute_fertility_static(inp, soil_prm)
    timings["soil_pass2"] = time.perf_counter() - t3
    report(0.5)
    suit = {}
    for p in STORED_PROFILES:
        s_, lim, _ = sl.suitability(st, p, alpha=prm.suitability_alpha)
        suit[p] = (s_, lim)
    # monthly consistency: one year from the static state with the climatology (G3)
    monthly_check = _monthly_check(st, inp)

    # -- 5.2.A primary deposits -----------------------------------------------------------------
    t4 = time.perf_counter()
    prospect = dep.tract_suitability(ctx)
    if not prm.include_p1:
        prospect.pop("R25", None)
    catalog = dep.plan_deposit_systems(ctx, prospect, prm.deposit_intensity)
    timings["deposits_primary"] = time.perf_counter() - t4
    report(0.7)
    # -- secondary ---------------------------------------------------------------------------------
    t5 = time.perf_counter()
    t_ann = inp.t_month.mean(axis=0)
    p_ann = inp.p_month.sum(axis=0)
    w_tr = sec.weathering_tracts(
        ctx, st["weathering_degree"], st["soil_age_yr"], t_ann, p_ann, fp > 0.4
    )
    coast = (
        (flat(H.get("coast_type", np.zeros(shape)), np.int64) > 0)
        if "coast_type" in H
        else np.zeros(n, bool)
    )
    wave = flat(H["wave_exposure"]) if "wave_exposure" in H else np.zeros(n)
    c_tr = sec.coastal_tracts(ctx, coast, wave)
    sys2, bod2, prov2 = sec.tract_systems(
        ctx, {**w_tr, **c_tr}, sec.TRACT_LAMBDA_SECONDARY, prm.deposit_intensity
    )
    river = flat(H["river_mask"], bool)
    sys3, bod3, placer_ledger = sec.river_placers(
        ctx, catalog, receiver, ctx.slope_m_km, lake, ocean, river, lay.spacing_km
    )
    felsic = np.isin(lith, [dep.LITH["arc_volcanic"], dep.LITH["granitoid_intrusion"]])
    sys4, bod4, basin_rows = sec.closed_basin_salts(
        ctx,
        H["lake_id"],
        H["endorheic_mask"],
        H["discharge_m3_s"],
        H["basin_id"],
        env["aridity_p_over_pet"],
        H.get("lake_depth_m", np.zeros(shape)),
        felsic,
    )
    catalog["systems"] += sys2 + sys3 + sys4
    catalog["bodies"] += bod2 + bod3 + bod4
    catalog["provinces"] += prov2
    catalog["systems"].sort(key=lambda s_: s_["deposit_id"])
    catalog["bodies"].sort(key=lambda b_: b_["body_id"])
    cls, qual = sec.construction_materials(ctx, inp.parent, depth, st["weathering_degree"])
    timings["deposits_secondary"] = time.perf_counter() - t5
    report(0.85)
    t6 = time.perf_counter()
    all_prospect = {**prospect, **w_tr, **c_tr}
    agg = dep.aggregate(catalog, ctx, all_prospect)
    timings["aggregate"] = time.perf_counter() - t6

    # -- layers ------------------------------------------------------------------------------------
    g = lambda a, dt=np.float32: np.asarray(a).reshape(shape).astype(dt)  # noqa: E731
    g3 = lambda a: np.asarray(a).reshape((a.shape[0],) + shape).astype(np.float32)  # noqa: E731
    bottoms = np.cumsum(np.nan_to_num(st["layer_thickness_m"]), axis=0)
    bottoms = np.where(present[None], bottoms, np.nan)
    layers = {
        "soil_age_yr": g(np.where(present, st["soil_age_yr"], np.nan)),
        "soil_layer_bottom_m": g3(bottoms),
        "soil_bulk_density_kg_m3": g3(st["bulk_density_kg_m3"]),
        "soil_soc_layers_kg_m2": g3(st["soc_layers_kg_m2"]),
        "soil_ph_layers": g3(st["ph_layers"]),
        "soil_soc_kg_m2": g(st["soc_kg_m2"]),
        "soil_organic_N_kg_m2": g(st["n_org_kg_m2"]),
        "soil_ammonium_N_kg_m2": g(st["nh4_kg_m2"]),
        "soil_nitrate_N_kg_m2": g(st["no3_kg_m2"]),
        "soil_N_net_mineralisation_kg_m2_yr": g(st["n_net_mineralisation_kg_m2_yr"]),
        "soil_P_primary_kg_m2": g(st["p_primary_kg_m2"]),
        "soil_P_labile_kg_m2": g(st["p_labile_kg_m2"]),
        "soil_P_sorbed_kg_m2": g(st["p_sorbed_kg_m2"]),
        "soil_P_organic_kg_m2": g(st["p_organic_kg_m2"]),
        "soil_P_occluded_kg_m2": g(st["p_occluded_kg_m2"]),
        "soil_K_mineral_kg_m2": g(st["k_primary_kg_m2"]),
        "soil_K_exchange_kg_m2": g(st["k_exchangeable_kg_m2"]),
        "soil_K_solution_kg_m2": g(st["k_solution_kg_m2"]),
        "soil_cec_cmolc_kg": g(st["cec_cmolc_kg"]),
        "soil_base_saturation_frac": g(st["base_saturation"]),
        "soil_ph": g(st["ph"]),
        "soil_acid_equivalent_molc_m2": g(st["acid_equivalent_molc_m2"]),
        "soil_buffer_capacity_molc_m2_per_ph": g(st["buffer_molc_m2_per_ph"]),
        "soil_carbonate_kg_m2": g(st["carbonate_kg_m2"]),
        "soil_salt_mass_kg_m2": g(st["salt_kg_m2"]),
        "soil_salinity_ec_ds_m": g(st["ec_ds_m"]),
        "soil_esp_pct": g(st["esp_pct"]),
        "soil_oxic_fraction": g(st["oxic_fraction"]),
        "soil_weathering_degree": g(st["weathering_degree"]),
        "soil_erosion_risk": g(st["erosion_risk"]),
        "soil_nutrient_supply_index": g(st["idx_nutrient"]),
        "soil_water_supply_index": g(st["idx_water"]),
        "soil_rooting_index": g(st["idx_rooting"]),
        "soil_fertility_potential": g(suit["native_generic"][0]),
        "soil_limiting_factor": g(suit["native_generic"][1], np.uint8),
        "geo_lithology_inferred": g(ctx.lith, np.uint8),
        "prospectivity_by_family": g3(agg["prospectivity_by_family"]),
        "resource_in_situ_t_by_family": g3(agg["in_situ_t_by_family"]),
        "deposit_count": g(agg["deposit_count"], np.int32),
        "shallowest_body_depth_m": g(agg["shallowest_body_depth_m"]),
        "surface_occurrence_mask": g(agg["surface_occurrence"], np.bool_),
        "overburden_m": g(agg["overburden_m"]),
        "resource_accessibility_proxy": g(agg["accessibility_proxy"]),
        "construction_material_class": g(cls, np.uint8),
        "construction_material_quality": g(qual),
    }
    for p in STORED_PROFILES:
        layers[f"suitability_{p}"] = g(suit[p][0])
    catalog["hex_offsets"] = agg["hex_offsets"]
    catalog["hex_bodies"] = agg["hex_bodies"]

    # -- summary -----------------------------------------------------------------------------------
    from collections import Counter

    by_model = Counter(s_["model_id"] for s_ in catalog["systems"])
    totals = {}
    for s_ in catalog["systems"]:
        for el, m in s_["contained_t"].items():
            totals[el] = totals.get(el, 0.0) + float(m)
    sp = present
    summary = {
        "algorithm_version": ALGORITHM_VERSION,
        "fertility_mode": prm.fertility_mode,
        "systems": len(catalog["systems"]),
        "bodies": len(catalog["bodies"]),
        "provinces": len(catalog["provinces"]),
        "eroded_systems": len(catalog["eroded"]),
        "systems_by_model": dict(sorted(by_model.items())),
        "in_situ_contained_t": {k: totals[k] for k in sorted(totals)},
        "surface_occurrences": int(sum(1 for s_ in catalog["systems"] if s_["surface_occurrence"])),
        "soil_columns": int(sp.sum()),
        "soil_soc_total_pg": float(np.nansum(st["soc_kg_m2"][sp]) * area_m2 / 1e12),
        "soil_ph_p05_p50_p95": [float(np.nanpercentile(st["ph"][sp], q)) for q in (5, 50, 95)]
        if sp.any()
        else None,
        "soil_fertility_mean": float(np.nanmean(suit["native_generic"][0][sp]))
        if sp.any()
        else None,
        "limiting_factor_shares": {
            f: float(np.mean(suit["native_generic"][1][sp] == k)) for k, f in enumerate(sl.FACTORS)
        }
        if sp.any()
        else {},
        "element_budget_over_soil_age": st["budget"],
        "erosion_transport_ledger": transport_ledger,
        "placer_ledger": placer_ledger,
        "closed_basins": basin_rows,
        "monthly_check": monthly_check,
        "unknown_inputs": catalog["unknown"],
        "families": list(FAMILIES),
        "lithology_shares": {
            dep.LITHOLOGY[k]: float(np.mean(ctx.lith[ctx.land] == k))
            for k in range(len(dep.LITHOLOGY))
            if ctx.land.any()
        },
    }
    provenance = {
        "soil_fertility": {
            "mode": prm.fertility_mode,
            "quality": "approximate",
            "carbon": "RothC 26.3 pools, analytic equilibrium (topsoil) + depth profile",
            "litter": "miami_npp_x_v50_cover"
            if "vegetation_fraction" in bio
            else "miami_npp_x_aet_proxy",
            "chemistry": "parent geochemistry table + weathering compartments [A]",
            "ph": "buffer curve from base saturation (proxy)",
            "ec": "salt mass / saturated pore water (proxy)",
        },
        "deposits": {
            "quality": "synthetic_not_calibrated",
            "lithology": "inferred_lithology",
            "process": "permissive tracts -> Poisson -> parametric bodies",
            "grade_tonnage": "joint log-normal, USGS-order medians (not calibrated)",
        },
        "oil_gas": "P1 simplified (no petroleum-system kinetics)" if prm.include_p1 else "off",
    }
    timings["total"] = time.perf_counter() - t0
    return {
        "layers": layers,
        "catalog": catalog,
        "summary": summary,
        "provenance": provenance,
        "timings": {k: round(v, 4) for k, v in timings.items()},
    }


def _monthly_check(st, inp) -> dict:
    """One explicit year of :func:`advance_soil_month` from the static state: element
    ledgers and the drift of the carbon stock (should stay small near equilibrium)."""
    present = np.asarray(st["soil_present"], bool)
    if not present.any():
        return {"status": "no_soil"}
    idx = np.nonzero(present)[0]
    if idx.size > 5000:
        idx = idx[:: max(1, idx.size // 5000)]
    s0 = {k: np.nan_to_num(v[idx]) for k, v in sl.state_from_static(st).items()}
    s = dict(s0)
    worst = {}
    clay = np.nan_to_num(inp.clay[idx]) * 100.0
    awc = np.maximum(np.nan_to_num(inp.awc_mm[idx]), 1.0)
    litter = np.nan_to_num(st["litter_input_kg_c_m2_yr"][idx])
    tree = np.clip(np.nan_to_num(inp.tree_fraction[idx]), 0, 1)
    theta = np.maximum(
        np.nan_to_num(inp.fc[idx]) * np.maximum(np.nan_to_num(inp.depth_m[idx]), 0.05), 0.005
    )
    for mth in range(12):
        deficit = np.maximum(awc - np.nan_to_num(inp.storage_month[mth, idx]), 0.0)
        month = {
            "t_c": inp.t_month[mth, idx],
            "moisture_mod": sl.rothc_moisture(deficit, 20 + 1.3 * awc),
            "cover_mod": np.where(np.nan_to_num(inp.vegetation_fraction[idx]) > 0.3, 0.6, 1.0),
            "anoxia": np.clip(np.nan_to_num(inp.waterlogging_months[idx]) / 12.0, 0, 1),
            "litter_c": litter / 12.0,
            "litter_cn": 25.0 + 25.0 * tree,
            "dpm_share": 0.2 * tree + 0.59 * (1 - tree),
            "clay_pct": clay,
            "drain_mm": np.nan_to_num(inp.percolation_mm_yr[idx]) / 12.0,
            "theta_d_m": theta,
            "uptake_n": np.zeros(idx.size),
            "uptake_p": np.zeros(idx.size),
            "uptake_k": np.zeros(idx.size),
            "n_fix": np.zeros(idx.size),
            "n_dep": np.full(idx.size, 5e-4 / 12.0),
            "salt_in": np.zeros(idx.size),
            "weather_p": np.full(idx.size, 2e-5),
            "weather_k": np.full(idx.size, 8e-6),
            "k_sorb": np.full(idx.size, 1.0),
            "k_desorb": np.full(idx.size, 0.15),
        }
        tot = sl._totals(s)
        s, rep = sl.advance_soil_month(s, month)
        for el, d in rep["ledger"].items():
            rel = np.abs(d["residual"]) / np.maximum(np.abs(tot[el]) + np.abs(d["in"]), 1e-12)
            worst[el] = max(worst.get(el, 0.0), float(np.max(rel)))
    c0 = s0["c_dpm"] + s0["c_rpm"] + s0["c_bio"] + s0["c_hum"] + s0["c_iom"]
    c1 = s["c_dpm"] + s["c_rpm"] + s["c_bio"] + s["c_hum"] + s["c_iom"]
    drift = np.abs(c1 - c0) / np.maximum(c0, 1e-6)
    return {
        "columns": int(idx.size),
        "max_relative_ledger_residual": worst,
        "carbon_drift_median": float(np.median(drift)),
        "carbon_drift_p95": float(np.percentile(drift, 95)),
    }


def run_v52(config, geometry, project, old_keys: dict, old_cache: dict, report):
    keys = resource_stage_keys(config, project)
    if old_keys.get("v52_resources") == keys["v52_resources"] and "v52_resources" in old_cache:
        data = old_cache["v52_resources"]
        return data, {"v52_resources": data}, ["v52_resources"]
    from .progress import sub_report

    report("v52_resources", 0.999)
    res = run_resources(
        config, project, report=lambda f: report(sub_report("v52_resources", f), 0.999)
    )
    warnings = []
    if res["catalog"]["unknown"]:
        warnings.append(
            "v5.2: inputs missing, the dependent deposit types are skipped: "
            + ", ".join(res["catalog"]["unknown"])
        )
    data = ResourceData(
        layers=res["layers"],
        catalog=res["catalog"],
        summary=res["summary"],
        provenance=res["provenance"],
        stage_keys=keys,
        warnings=warnings,
        timings=res["timings"],
    )
    return data, {"v52_resources": data}, []
