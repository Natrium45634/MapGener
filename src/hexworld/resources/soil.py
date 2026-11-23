"""Soil fertility of TZ 5.2 part II: separate stocks and limits, not one "greenness" number.

Everything works on flat arrays of land hexes (or synthetic columns in the tests). Stocks
are kg of the element per m2 of horizontal area over the whole soil profile; ``top`` values
refer to the 0-0.3 m layer. Formulas marked [I] come from a published source, [A] are the
HexWorld adaptation with tunable coefficients (``SoilChemParams``); see
``docs/research/v5_2_sources.md``.

* fine-earth mass of a layer ``M = D rho (1 - f_coarse)``; ``rho`` is the bulk density of
  the fine earth (so the stones are removed once, TZ 5.1 / N06);
* SOC: RothC-type pools DPM, RPM, BIO, HUM + IOM [I: Coleman & Jenkinson, RothC 26.3]:
  monthly rate modifiers a(T), b(moisture), c(cover) and an anoxia factor [A]; equilibrium
  is analytic, a young soil approaches it as ``1 - exp(-k t)``;
* litter input from the Miami NPP model [I: Lieth 1975] times the v5.0 vegetation fraction
  (provenance ``miami_npp_x_v50_cover``; the v5.0 productivity is an index, not NPP);
* P (primary, labile, sorbed, organic, occluded), K (primary, exchangeable, solution) and
  base cations (primary, exchangeable, solution) as linear compartment systems integrated
  over the soil age with implicit Euler on geometric steps: every transfer keeps the
  element, leaching and erosion are booked as exports (exact ledger);
* pH from the base saturation (a buffer curve, quality ``proxy``) and the carbonate left;
  ``delta_pH = -delta_acid / B`` with the buffer capacity B (TZ 9);
* salts as a mass with inputs and leaching (evaporation never removes salt); EC from the
  salt mass and the soil water at saturation (proxy);
* suitability per profile: ``S = min(s)^alpha * (prod s^w)^(1 - alpha)`` (TZ 11) with the
  whole vector of factors and the dominant limitation kept.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# -- parent material chemistry --------------------------------------------------------------
#: Parent material geochemistry [A]: total P (mg/kg), total K (mass fraction), weatherable
#: base cations Ca+Mg (cmolc/kg of fine earth), carbonate (CaCO3 mass fraction), share of
#: 2:1 clay minerals of the fresh material, relative mineral reactivity, primary soluble
#: salt (kg per t of fine earth). Ranges of common rocks (granite P 500-900 ppm, K 3-4 %;
#: basalt P 1000-1500 ppm, K 0.5-1 %; shale K 2.5-3 %; loess carbonate 5-20 %).
PARENT_CHEM = {
    "none": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "crystalline_bedrock": (650.0, 0.032, 25.0, 0.0, 0.4, 0.6, 0.0),
    "sedimentary_bedrock": (600.0, 0.020, 40.0, 0.04, 0.6, 0.8, 0.05),
    "orogenic_bedrock": (700.0, 0.025, 35.0, 0.01, 0.5, 0.7, 0.0),
    "young_volcanic": (950.0, 0.012, 90.0, 0.0, 0.2, 2.5, 0.0),
    "weathered_volcanic": (1200.0, 0.007, 110.0, 0.0, 0.3, 1.0, 0.0),
    "alluvium": (750.0, 0.018, 55.0, 0.02, 0.6, 1.0, 0.05),
    "glacial_till": (700.0, 0.021, 50.0, 0.02, 0.6, 1.1, 0.0),
    "aeolian_sand": (220.0, 0.008, 8.0, 0.01, 0.3, 0.5, 0.05),
    "loess": (700.0, 0.018, 70.0, 0.10, 0.7, 1.2, 0.05),
    "marine_lacustrine": (800.0, 0.022, 60.0, 0.06, 0.8, 0.9, 1.0),
    "colluvium": (650.0, 0.022, 35.0, 0.01, 0.5, 0.9, 0.0),
}
PARENT_ORDER = tuple(PARENT_CHEM)
#: materials that arrive already weathered (partly saturated exchange complex)
TRANSPORTED = (
    "alluvium",
    "glacial_till",
    "loess",
    "marine_lacustrine",
    "colluvium",
    "aeolian_sand",
)

KG_PER_MOLC_BC = 0.017  # Ca/Mg mix: ~17 g per mol of charge
KG_PER_MOLC_K = 0.0391
PARTICLE_DENSITY = 2650.0  # kg/m3, mineral particles (not for peat: Adams mixing below)
OM_PARTICLE_BD = 244.0  # kg/m3, Adams (1973) bulk density of pure organic matter
OC_TO_OM = 1.724  # van Bemmelen factor
SECONDS_YEAR = 365.25 * 86400.0
#: functional layers (top, sub, deep): top to 0.3 m, sub to 1.0 m, deep to the soil depth
LAYER_BOTTOMS_M = (0.3, 1.0)

#: RothC 26.3 decomposition rates (1/yr) [I]
K_DPM, K_RPM, K_BIO, K_HUM = 10.0, 0.3, 0.66, 0.02
#: C:N of the pools [A] (litter by cover; microbes / humus ~ 8-11)
CN_BIO, CN_HUM, CN_IOM = 8.0, 11.0, 15.0


@dataclass(frozen=True)
class SoilChemParams:
    """Tunable coefficients of the fertility model [A] (units in the names)."""

    alpha: float = 0.5  # limiting-factor weight of the suitability (TZ 11)
    k_weather_p: float = 2.0e-5  # 1/yr apatite weathering at 15 degC, moderate leaching
    k_weather_k: float = 8.0e-6  # 1/yr K-bearing silicates (micas, feldspars)
    k_weather_bc: float = 3.0e-5  # 1/yr Ca/Mg silicates at 15 degC, 300 mm/yr drainage
    activation_kj_mol: float = 60.0  # Arrhenius activation energy of silicate weathering
    k_sorb: float = 1.0  # 1/yr labile -> sorbed at empty sorption sites
    k_desorb: float = 0.15  # 1/yr sorbed -> labile
    k_occlude: float = 2.5e-5  # 1/yr sorbed -> occluded at full oxide development
    p_solution_share: float = 0.002  # share of labile P dissolved in the soil water
    k_bio_cycle: float = 0.6  # 1/yr labile P -> organic P at full productivity
    dust_p_kg_m2_yr: float = 1.0e-6  # atmospheric P (0.01 kg/ha/yr)
    n_deposition_kg_m2_yr: float = 5.0e-4  # 5 kg N/ha/yr
    n_fixation_max_kg_m2_yr: float = 2.0e-3  # biological fixation at full productivity
    k_nitrify: float = 20.0  # 1/yr NH4 -> NO3, aerobic, 15 degC
    k_uptake: float = 30.0  # 1/yr mineral N uptake at full cover
    k_denit: float = 8.0  # 1/yr NO3 -> N2/N2O in fully anoxic soil
    marine_salt_kg_m2_yr: float = 4.0e-3  # aerosol salt at the coast (40 kg/ha/yr)
    endorheic_salt_kg_m2_yr: float = 0.05  # solute delivery to the floor of a closed basin
    salt_deflation_per_yr: float = 2.0e-5  # wind removal of salt crusts
    ph_floor: float = 4.2  # buffer curve pH = floor + span BS^power (Al buffering below)
    ph_span: float = 2.8
    ph_power: float = 0.9
    acid_conc_molc_m3: float = 0.3  # carbonic + organic acids of percolating water
    bc_deposition_molc_m2_yr: float = 0.02  # dust + sea-salt base cations (0.2 keq/ha/yr)
    carbonate_solubility_kg_m3: float = 0.1  # CaCO3 removed per m3 of percolating water
    integration_steps: int = 48
    max_soil_age_yr: float = 4.0e6
    min_soil_age_yr: float = 100.0


# -- exact reference formulas (TZ 5.2 sec. 23) ----------------------------------------------------
def fine_earth_mass(depth_m, bulk_density_kg_m3, coarse_frac):
    """kg of fine earth per m2: ``D rho (1 - f_coarse)`` (rho of the fine earth)."""
    return np.asarray(depth_m) * np.asarray(bulk_density_kg_m3) * (1.0 - np.asarray(coarse_frac))


def soc_stock(depth_m, bulk_density_kg_m3, c_frac, coarse_frac):
    """kg C/m2 (N01: 0.30 m, 1300 kg/m3, 2 %, 10 % stones -> 7.02 kg C/m2 = 70.2 t C/ha)."""
    return fine_earth_mass(depth_m, bulk_density_kg_m3, coarse_frac) * np.asarray(c_frac)


def element_mass(depth_m, bulk_density_kg_m3, coarse_frac, mass_frac):
    """kg/m2 of an element given as a mass fraction of the fine earth (N06)."""
    return fine_earth_mass(depth_m, bulk_density_kg_m3, coarse_frac) * np.asarray(mass_frac)


def kg_m2_to_t_ha(x):
    return np.asarray(x) * 10.0


def awc_mm(depth_m, fc, pwp, coarse_frac=0.0, theta_of="fine_earth"):
    """AWC = 1000 D (FC - PWP) (1 - f_coarse); no stone factor if theta refers to the whole
    soil volume (N02: 1 m, 0.30 / 0.12 -> 180 mm)."""
    f = (1.0 - np.asarray(coarse_frac)) if theta_of == "fine_earth" else 1.0
    return 1000.0 * np.asarray(depth_m) * np.maximum(np.asarray(fc) - np.asarray(pwp), 0.0) * f


def first_order_decay(c, k_per_yr, dt_yr):
    """Analytic step of linear decay without input; returns (remaining, decomposed) (N03)."""
    rem = np.asarray(c) * np.exp(-np.asarray(k_per_yr) * dt_yr)
    return rem, np.asarray(c) - rem


def leaching_flux(q_mm_yr, conc_kg_m3, available_kg_m2=None, dt_yr=1.0):
    """kg/(m2 yr) carried by drainage ``q`` at concentration ``c`` (N04), limited to the
    stock if one is given."""
    f = np.asarray(q_mm_yr) / 1000.0 * np.asarray(conc_kg_m3)
    if available_kg_m2 is not None:
        f = np.minimum(f, np.asarray(available_kg_m2) / dt_yr)
    return f


def ph_change(delta_acid_molc_m2, buffer_molc_m2_per_ph, max_abs=2.0):
    """Bounded ``delta_pH = -delta_acid / B`` (N05; quality ``proxy``)."""
    b = np.maximum(np.asarray(buffer_molc_m2_per_ph, dtype=np.float64), 1e-9)
    return np.clip(-np.asarray(delta_acid_molc_m2) / b, -max_abs, max_abs)


def ph_from_base_saturation(bs, prm=None):
    """Buffer curve pH = floor + span BS^power [A]: BS 0.1 -> 4.6, 0.5 -> 5.7, 1 -> 7.0
    (Al buffering keeps strongly leached soils near 4.2); quality ``proxy``."""
    prm = prm or SoilChemParams()
    return (
        prm.ph_floor
        + prm.ph_span * np.clip(np.asarray(bs, dtype=np.float64), 0.0, 1.0) ** prm.ph_power
    )


def dph_dbs(bs, prm=None):
    prm = prm or SoilChemParams()
    b = np.clip(np.asarray(bs, dtype=np.float64), 1e-3, 1.0)
    return prm.ph_span * prm.ph_power * b ** (prm.ph_power - 1.0)


def aggregate_ph(ph, weights):
    """Area / volume aggregate of pH through the H+ activity, never a linear mean:
    ``-log10(sum w 10^-pH / sum w)``."""
    ph = np.asarray(ph, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    return float(-np.log10(np.sum(w * 10.0 ** (-ph)) / np.sum(w)))


def porosity(bulk_density_kg_m3, particle_density=PARTICLE_DENSITY):
    return np.clip(1.0 - np.asarray(bulk_density_kg_m3) / particle_density, 0.0, 1.0)


def bulk_density_adams(mineral_bd_kg_m3, om_percent):
    """Adams (1973) mixing: ``100 / (OM / 244 + (100 - OM) / BD_mineral)`` [I]."""
    om = np.clip(np.asarray(om_percent, dtype=np.float64), 0.0, 100.0)
    return 100.0 / (om / OM_PARTICLE_BD + (100.0 - om) / np.asarray(mineral_bd_kg_m3))


def miami_npp_kg_c(t_c, p_mm):
    """Miami model NPP (Lieth 1975) in kg C/(m2 yr): min of the T and P limits of
    g dry matter / m2 / yr, times 0.45 C [I]."""
    t = np.asarray(t_c, dtype=np.float64)
    p = np.maximum(np.asarray(p_mm, dtype=np.float64), 0.0)
    npp_t = 3000.0 / (1.0 + np.exp(1.315 - 0.119 * t))
    npp_p = 3000.0 * (1.0 - np.exp(-0.000664 * p))
    return 0.45e-3 * np.minimum(npp_t, npp_p)


def rothc_temperature(t_c):
    """RothC rate modifier a(T) = 47.91 / (1 + exp(106.06 / (T + 18.27))), 0 below -18 degC."""
    t = np.asarray(t_c, dtype=np.float64)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        a = 47.91 / (1.0 + np.exp(106.06 / (t + 18.27)))
    return np.where(t > -18.0, np.nan_to_num(a), 0.0)


def rothc_moisture(deficit_mm, max_deficit_mm):
    """RothC b: 1 while the topsoil moisture deficit is below 0.444 of the maximum, then
    linearly to 0.2 at the maximum deficit."""
    d = np.asarray(deficit_mm, dtype=np.float64)
    m = np.maximum(np.asarray(max_deficit_mm, dtype=np.float64), 1e-9)
    b = 0.2 + 0.8 * (m - d) / (m - 0.444 * m)
    return np.where(d < 0.444 * m, 1.0, np.clip(b, 0.2, 1.0))


def rothc_humification(clay_pct):
    """Share of decomposed C that becomes BIO + HUM: 1 / (1 + x),
    x = 1.67 (1.85 + 1.60 exp(-0.0786 clay%)) (CO2 / (BIO + HUM)) [I]."""
    x = 1.67 * (1.85 + 1.60 * np.exp(-0.0786 * np.asarray(clay_pct, dtype=np.float64)))
    return 1.0 / (1.0 + x)


def falloon_iom(soc_kg_m2):
    """Inert organic matter IOM = 0.049 SOC^1.139 (t C/ha in the source; kg/m2 here via
    the 10x conversion) [I: Falloon et al. 1998]."""
    soc_tha = np.maximum(np.asarray(soc_kg_m2, dtype=np.float64), 0.0) * 10.0
    return 0.049 * soc_tha**1.139 / 10.0


def rothc_equilibrium(input_kg_c, dpm_share, clay_pct, rate_modifier, loss_per_yr=0.0):
    """Analytic RothC steady state (kg C/m2) for a constant input and mean modifier ``m``,
    with an optional extra loss ``L`` of every pool (erosion, 1/yr).

    DPM* = I_d / (k_d m + L), RPM* = I_r / (k_r m + L); every decomposed flux returns a
    share ``e`` as BIO (46 %) and HUM (54 %): a 2 x 2 linear system for BIO and HUM.
    Without loss the total decomposition flux is ``I / (1 - e)``.
    """
    m = np.maximum(np.asarray(rate_modifier, dtype=np.float64), 1e-6)
    i = np.maximum(np.asarray(input_kg_c, dtype=np.float64), 0.0)
    loss = np.asarray(loss_per_yr, dtype=np.float64)
    e = rothc_humification(clay_pct)
    dpm = i * dpm_share / (K_DPM * m + loss)
    rpm = i * (1.0 - dpm_share) / (K_RPM * m + loss)
    s = K_DPM * m * dpm + K_RPM * m * rpm
    kb, kh = K_BIO * m, K_HUM * m
    a11 = kb + loss - 0.46 * e * kb
    a12 = -0.46 * e * kh
    a21 = -0.54 * e * kb
    a22 = kh + loss - 0.54 * e * kh
    det = a11 * a22 - a12 * a21
    bio = (0.46 * e * s * a22 - a12 * 0.54 * e * s) / det
    hum = (a11 * 0.54 * e * s - a21 * 0.46 * e * s) / det
    co2 = (1.0 - e) * (s + kb * bio + kh * hum)
    return {"dpm": dpm, "rpm": rpm, "bio": bio, "hum": hum, "co2_flux": co2}


# -- linear compartment systems -----------------------------------------------------------------
def integrate_linear(transfer, loss, inputs, x0, t_yr, steps=48, span=1.0e4):
    """Integrate ``dx/dt = T x - diag(loss) x + b`` over ``t_yr`` per hex (implicit Euler on
    geometric steps, the same fractions of each hex's own time).

    ``transfer``: (n, k, k) with ``T[i, j]`` the rate (1/yr) from pool j to pool i
    (diagonal ignored: the outflow of j is the column sum). ``loss``: (n, k) export rates;
    ``inputs``: (n, k) kg/(m2 yr). Returns ``(x, exported, added)``; the ledger
    ``sum(x0) + added == sum(x) + exported`` holds to round-off for every hex.
    """
    t = np.asarray(transfer, dtype=np.float64).copy()
    n, k, _ = t.shape
    idx = np.arange(k)
    t[:, idx, idx] = 0.0
    out = t.sum(axis=1)  # (n, k) outflow rate of each pool
    a = t.copy()
    a[:, idx, idx] = -(out + loss)
    x = np.asarray(x0, dtype=np.float64).copy()
    b = np.asarray(inputs, dtype=np.float64)
    loss = np.asarray(loss, dtype=np.float64)
    exported = np.zeros(n)
    added = np.zeros(n)
    # geometric steps whose ratio shrinks with more steps (the largest step ~ ln(span) /
    # steps of the time): the scheme converges as the steps are refined (test S08)
    ratio = span ** (1.0 / max(steps - 1, 1))
    w = ratio ** np.arange(steps)
    frac = w / w.sum()
    eye = np.eye(k)[None]
    tt = np.asarray(t_yr, dtype=np.float64)
    for f in frac:
        dt = (f * tt)[:, None, None]
        m = eye - dt * a
        rhs = x + dt[:, :, 0] * b
        x = np.linalg.solve(m, rhs[..., None])[..., 0]
        x = np.maximum(x, 0.0)
        exported += (f * tt) * (loss * x).sum(axis=1)
        added += (f * tt) * b.sum(axis=1)
    return x, exported, added


# -- the static fertility state (B0-B4) ------------------------------------------------------------
@dataclass
class FertilityInputs:
    """Flat arrays over the soil columns (land hexes); NaN-free where ``soil_present``."""

    soil_present: np.ndarray
    parent: np.ndarray  # index into PARENT_ORDER
    sand: np.ndarray
    clay: np.ndarray
    coarse: np.ndarray
    depth_m: np.ndarray
    porosity_mineral: np.ndarray  # saturated water content of v4.5 (mineral soil)
    fc: np.ndarray
    pwp: np.ndarray
    t_month: np.ndarray  # (12, n) degC
    p_month: np.ndarray  # (12, n) mm
    pet_month: np.ndarray  # (12, n) mm
    storage_month: np.ndarray  # (12, n) plant-available water, mm
    awc_mm: np.ndarray
    percolation_mm_yr: np.ndarray  # drainage below the root zone (recharge)
    waterlogging_months: np.ndarray
    vegetation_fraction: np.ndarray
    tree_fraction: np.ndarray
    soil_erosion_m_yr: np.ndarray  # long-term surface lowering (v4 rate)
    soil_age_yr: np.ndarray
    marine_influence: np.ndarray  # 0..1
    endorheic_floor: np.ndarray  # bool
    carbonate_extra: np.ndarray = field(default=None)  # extra carbonate from lithology
    deposition_p_kg_m2_yr: np.ndarray = field(default=None)  # alluvial / tephra inputs
    deposition_k_kg_m2_yr: np.ndarray = field(default=None)
    deposition_bc_kg_m2_yr: np.ndarray = field(default=None)
    deposition_c_kg_m2_yr: np.ndarray = field(default=None)
    ash_stress: np.ndarray = field(default=None)  # 0..1 v5.1 volcanic ash stress
    tephra_m: np.ndarray = field(default=None)


def _arrhenius(t_c, ea_kj):
    t = np.asarray(t_c, dtype=np.float64) + 273.15
    return np.exp(ea_kj * 1000.0 / 8.314 * (1.0 / 288.15 - 1.0 / np.maximum(t, 200.0)))


def layer_thickness(depth_m):
    """(3, n) thickness of the top / sub / deep functional layers (m)."""
    d = np.maximum(np.asarray(depth_m, dtype=np.float64), 0.0)
    a = np.minimum(d, LAYER_BOTTOMS_M[0])
    b = np.minimum(d, LAYER_BOTTOMS_M[1]) - a
    c = d - a - b
    return np.stack([a, b, c])


def soc_depth_shares(depth_m, tree_fraction):
    """(3, n) shares of the profile SOC in the three layers: exponential decline with
    ``z0`` 0.35 m (grass) .. 0.25 m (forest) [A, after Jobbagy & Jackson 2000]."""
    thick = layer_thickness(depth_m)
    z0 = 0.35 - 0.10 * np.clip(tree_fraction, 0.0, 1.0)
    tops = np.cumsum(np.vstack([np.zeros_like(thick[0]), thick[:-1]]), axis=0)
    bots = tops + thick
    w = np.exp(-tops / z0) - np.exp(-bots / z0)
    tot = np.maximum(w.sum(axis=0), 1e-12)
    return w / tot


def compute_fertility_static(inp: FertilityInputs, prm: SoilChemParams | None = None) -> dict:
    """Static (quasi-equilibrium) fertility state of every column; NoData where no soil.

    Returns a dict of flat arrays (stocks, chemistry, indices) plus ``budget`` (element
    ledgers: initial + added = final + exported per element, worst relative residual).
    """
    prm = prm or SoilChemParams()
    n = inp.depth_m.size
    present = np.asarray(inp.soil_present, dtype=bool)
    chem = np.array([PARENT_CHEM[k] for k in PARENT_ORDER])[np.asarray(inp.parent, int)]
    p_ppm, k_frac, bc_cmol, carb, smect, react, salt_pt = chem.T
    depth = np.where(present, np.maximum(inp.depth_m, 0.0), 0.0)
    coarse = np.clip(np.nan_to_num(inp.coarse), 0.0, 0.9)
    clay = np.clip(np.nan_to_num(inp.clay, nan=0.2), 0.0, 1.0)
    sand = np.clip(np.nan_to_num(inp.sand, nan=0.4), 0.0, 1.0)
    t12 = np.asarray(inp.t_month, dtype=np.float64)
    p12 = np.asarray(inp.p_month, dtype=np.float64)
    t_ann = t12.mean(axis=0)
    p_ann = p12.sum(axis=0)
    age = np.clip(
        np.nan_to_num(inp.soil_age_yr, nan=prm.min_soil_age_yr),
        prm.min_soil_age_yr,
        prm.max_soil_age_yr,
    )
    veg = np.clip(np.nan_to_num(inp.vegetation_fraction), 0.0, 1.0)
    tree = np.clip(np.nan_to_num(inp.tree_fraction), 0.0, 1.0)
    perc = np.maximum(np.nan_to_num(inp.percolation_mm_yr), 0.0)
    q = perc / 1000.0  # m/yr
    fc = np.nan_to_num(inp.fc, nan=0.25)
    pwp = np.nan_to_num(inp.pwp, nan=0.1)
    theta_d = np.maximum(fc * np.maximum(depth, 0.05), 0.005)  # m of water at FC in the profile
    erosion = np.maximum(np.nan_to_num(inp.soil_erosion_m_yr), 0.0)
    k_eros = np.where(depth > 0.0, erosion / np.maximum(depth, 0.05), 0.0)

    # -- organic carbon (RothC) ------------------------------------------------------------
    npp = miami_npp_kg_c(t_ann, p_ann) * veg
    awc = np.maximum(np.nan_to_num(inp.awc_mm), 1.0)
    deficit = np.maximum(awc[None] - np.nan_to_num(inp.storage_month), 0.0)
    max_def = 20.0 + 1.3 * awc  # RothC maximum topsoil deficit scale [A for the full AWC]
    a = rothc_temperature(t12)
    b = rothc_moisture(deficit, max_def[None])
    c = np.where(veg[None] > 0.3, 0.6, 1.0)
    wl = np.clip(np.nan_to_num(inp.waterlogging_months) / 12.0, 0.0, 1.0)
    anoxia = 1.0 - 0.85 * wl  # saturated months decompose at 15 % [A]
    m_rate = np.maximum((a * b * c).mean(axis=0) * anoxia, 1e-4)
    dpm_share = 0.2 * tree + 0.59 * (1.0 - tree)  # DPM/RPM 0.25 forest, 1.44 grass [I]
    c_in_dep = (
        np.zeros(n)
        if inp.deposition_c_kg_m2_yr is None
        else np.maximum(inp.deposition_c_kg_m2_yr, 0.0)
    )
    c_input = npp + c_in_dep
    # RothC is calibrated on the topsoil (0-0.23 / 0.30 m): the pools describe the top
    # layer, the profile follows the depth distribution of SOC [A]
    shares = soc_depth_shares(depth, tree)
    to_profile = 1.0 / np.maximum(shares[0], 0.05)
    eq = rothc_equilibrium(c_input, dpm_share, 100.0 * clay, m_rate, k_eros)
    approach = 1.0 - np.exp(-K_HUM * m_rate * age)  # humus pool still building in young soils
    eq = {k_: v * to_profile for k_, v in eq.items()}
    hum = eq["hum"] * approach
    soc_bio = eq["dpm"] + eq["rpm"] + eq["bio"] + hum
    iom = falloon_iom(soc_bio) * np.clip(age / 1.0e4, 0.0, 1.0)
    soc = soc_bio + iom
    soc = np.where(present, soc, np.nan)

    # -- bulk density with organic matter ------------------------------------------------------
    thick = layer_thickness(depth)
    min_bd = (
        1.0 - np.clip(np.nan_to_num(inp.porosity_mineral, nan=0.45), 0.2, 0.8)
    ) * PARTICLE_DENSITY
    min_bd = np.clip(min_bd, 900.0, 1800.0)
    bd_layers = np.empty((3, n))
    soc_layers = np.empty((3, n))
    for j in range(3):
        cl = np.nan_to_num(soc) * shares[j]
        mass_guess = np.maximum(thick[j] * min_bd * (1.0 - coarse), 1e-9)
        om_pct = np.clip(100.0 * OC_TO_OM * cl / mass_guess, 0.0, 95.0)
        for _ in range(3):  # BD and OM% depend on each other (the mass shrinks with OM)
            bd = bulk_density_adams(min_bd, om_pct)
            mass = np.maximum(thick[j] * bd * (1.0 - coarse), 1e-9)
            om_pct = np.clip(100.0 * OC_TO_OM * cl / mass, 0.0, 95.0)
        bd_layers[j] = bd
        soc_layers[j] = cl
    fe_mass = (thick * bd_layers * (1.0 - coarse)[None]).sum(axis=0)  # kg fine earth / m2
    top_mass = np.maximum(thick[0] * bd_layers[0] * (1.0 - coarse), 1e-9)
    oc_top_frac = soc_layers[0] / top_mass

    # -- weathering environment ----------------------------------------------------------------
    f_t = _arrhenius(t_ann, prm.activation_kj_mol)
    f_w = np.clip(np.sqrt(perc / 300.0), 0.1, 2.0)  # leaching water drives weathering [A]
    weather_env = f_t * f_w * react
    oxide = 1.0 - np.exp(-prm.k_weather_bc * weather_env * age)  # weathering degree 0..1

    # -- phosphorus: primary, labile, sorbed, organic, occluded --------------------------------
    p_tot0 = fe_mass * p_ppm * 1e-6
    carb_now = (
        carb + (0.0 if inp.carbonate_extra is None else np.nan_to_num(inp.carbonate_extra))
    ) > 0.0
    # Fe/Al oxides of weathered soils and Ca phosphates of calcareous soils both fix P
    sorption = np.clip(0.3 + 1.2 * clay + 1.5 * oxide + 2.0 * carb_now, 0.2, 4.0)
    prod = np.clip(npp / 0.6, 0.0, 1.0)
    tp = np.zeros((n, 5, 5))
    tp[:, 1, 0] = prm.k_weather_p * weather_env  # primary -> labile
    tp[:, 2, 1] = prm.k_sorb * sorption  # labile -> sorbed
    tp[:, 1, 2] = prm.k_desorb  # sorbed -> labile
    tp[:, 3, 1] = prm.k_bio_cycle * prod  # labile -> organic (plant / microbial cycle)
    tp[:, 1, 3] = 0.05 * m_rate  # organic -> labile (mineralisation with the C turnover)
    tp[:, 4, 2] = prm.k_occlude * oxide * f_w  # sorbed -> occluded by Fe/Al oxides
    lp = np.zeros((n, 5))
    lp[:, 1] = q / theta_d * prm.p_solution_share
    lp += k_eros[:, None]
    bp = np.zeros((n, 5))
    bp[:, 0] = prm.dust_p_kg_m2_yr
    if inp.deposition_p_kg_m2_yr is not None:
        bp[:, 0] += np.maximum(inp.deposition_p_kg_m2_yr, 0.0)
    x0 = np.zeros((n, 5))
    x0[:, 0] = p_tot0
    xp, p_exp, p_add = integrate_linear(tp, lp, bp, x0, age, prm.integration_steps)

    # -- base cations and potassium: primary minerals and the exchange complex -------------
    om_frac = np.nan_to_num(soc) * OC_TO_OM / np.maximum(fe_mass, 1e-9)
    # CEC (cmolc/kg): clay minerals lose charge as kaolinite / oxides form; organic matter
    # 200 cmolc/kg at pH 7 [A after Parfitt et al. 1995]
    cec_clay = (10.0 + 70.0 * smect * (1.0 - oxide)) * (1.0 - 0.5 * oxide)
    cec_ph7 = clay * cec_clay + 200.0 * om_frac
    cec_ph7 = np.maximum(cec_ph7, 0.5)
    cec_molc = cec_ph7 / 100.0 * fe_mass  # molc/m2 (cmolc/kg x kg / 100)
    # primary minerals decay first-order: M(t) = M0 exp(-k t) (analytic), the release
    # W = k M(t) feeds the exchange complex; steady base saturation of the flux balance
    # BS = W / (W + A) with the acid load A = q c_acid (carbonic + organic acids) [A]
    k_bc = prm.k_weather_bc * weather_env + k_eros
    k_k = prm.k_weather_k * weather_env + k_eros
    bc_tot0 = fe_mass * bc_cmol / 100.0  # molc/m2
    k_tot0 = fe_mass * k_frac / KG_PER_MOLC_K  # molc/m2
    bc_prim = bc_tot0 * np.exp(-k_bc * age)
    k_prim = k_tot0 * np.exp(-k_k * age)
    w_bc = prm.k_weather_bc * weather_env * bc_prim
    w_k = prm.k_weather_k * weather_env * k_prim
    c_acid = prm.acid_conc_molc_m3 * (1.0 + tree)
    acid_load = q * c_acid + 1e-6
    dep_bc = prm.bc_deposition_molc_m2_yr * (
        1.0 + 2.0 * np.clip(np.nan_to_num(inp.marine_influence), 0, 1)
    )
    supply = w_bc + w_k + dep_bc
    bs_ss = supply / (supply + acid_load)
    # relaxation from the fresh material's saturation over the exchange turnover time
    bs0 = np.where(np.isin(inp.parent, [PARENT_ORDER.index(n_) for n_ in TRANSPORTED]), 0.7, 1.0)
    tau = cec_molc / (supply + acid_load)
    bs = bs_ss + (bs0 - bs_ss) * np.exp(-age / np.maximum(tau, 1e-9))
    # carbonate: zero-order dissolution by percolating water until exhausted
    carb_frac = carb + (0.0 if inp.carbonate_extra is None else np.nan_to_num(inp.carbonate_extra))
    carb0 = fe_mass * carb_frac
    carb_left = np.maximum(carb0 - q * prm.carbonate_solubility_kg_m3 * age, 0.0)
    bs = np.clip(np.where(carb_left > 0.0, 1.0, bs), 0.0, 1.0)
    k_share = np.where(supply > 0, w_k / np.maximum(supply, 1e-30), 0.1)
    k_share = np.clip(np.where(supply > 1e-12, k_share, 0.05), 0.02, 0.12)
    exch_total = bs * cec_molc
    exch_k_molc = np.minimum(exch_total * k_share, k_tot0 - k_prim + dep_bc * 0.05 * age)
    exch_k_molc = np.maximum(exch_k_molc, 0.0)
    exch_bc_molc = np.minimum(
        exch_total - exch_k_molc,
        np.maximum(bc_tot0 - bc_prim, 0.0) + dep_bc * 0.95 * age + (carb0 - carb_left) / 0.05,
    )
    k_sol_molc = 0.003 * exch_k_molc
    xk = np.stack([k_prim, exch_k_molc, k_sol_molc], axis=1) * KG_PER_MOLC_K
    # ledgers: deposition and dissolved carbonate are the inputs, leaching the residual
    k_add = dep_bc * 0.05 * age * KG_PER_MOLC_K
    k_exp = k_tot0 * KG_PER_MOLC_K + k_add - xk.sum(axis=1)
    xb = np.stack([bc_prim, exch_bc_molc], axis=1) * KG_PER_MOLC_BC
    carb_molc = (carb0 - carb_left) / 0.05  # CaCO3: 50 g per mol of charge
    bc_add = (dep_bc * 0.95 * age + carb_molc) * KG_PER_MOLC_BC
    bc_exp = bc_tot0 * KG_PER_MOLC_BC + bc_add - xb.sum(axis=1)
    k_tot0 = k_tot0 * KG_PER_MOLC_K
    bc_tot0 = bc_tot0 * KG_PER_MOLC_BC

    # -- salts ------------------------------------------------------------------------------
    salt_in = prm.marine_salt_kg_m2_yr * np.clip(
        np.nan_to_num(inp.marine_influence), 0, 1
    ) + prm.endorheic_salt_kg_m2_yr * np.asarray(inp.endorheic_floor, dtype=float)
    salt0 = fe_mass * salt_pt * 1e-3
    # removal: drainage, erosion and wind deflation of surface crusts (2e-5 /yr) [A]
    k_salt = q / theta_d + k_eros + prm.salt_deflation_per_yr
    salt = np.where(
        k_salt * age > 1e-9,
        salt0 * np.exp(-k_salt * age)
        + salt_in / np.maximum(k_salt, 1e-12) * (1.0 - np.exp(-k_salt * age)),
        salt0 + salt_in * age,
    )
    sat_water_m3 = np.maximum(
        np.nan_to_num(inp.porosity_mineral, nan=0.45) * np.maximum(depth, 0.05), 1e-3
    )
    # only the dissolved part counts for EC: above saturation the salt is solid
    tds_mg_l = np.minimum(salt / sat_water_m3, 360.0) * 1000.0
    ec = tds_mg_l / 640.0
    # sodicity: marine / closed-basin salts are Na-rich; leaching keeps the exchange clean
    esp = np.clip(100.0 * salt / np.maximum(salt + 0.02 * cec_molc, 1e-9), 0.0, 60.0)

    # -- pH and buffering ---------------------------------------------------------------------
    ph = ph_from_base_saturation(bs, prm)
    ph = np.where(carb_left > 0.0, np.maximum(ph, 7.9), ph)
    ph = np.where(esp > 15.0, np.maximum(ph, 8.3 + 0.02 * (esp - 15.0)), ph)
    ash = 0.0 if inp.ash_stress is None else np.clip(np.nan_to_num(inp.ash_stress), 0.0, 1.0)
    acid_eq = cec_molc * (1.0 - bs)
    buffer = cec_molc / dph_dbs(bs, prm) + 2.0 * carb_left / 0.05
    buffer = np.maximum(buffer, 0.05 * cec_molc + 1e-3)
    # sulfuric acid of fresh ash: an acid load of 0.5 molc/m2 at full stress, buffered
    ph = ph + ph_change(0.5 * ash, buffer)
    ph = np.clip(ph, 3.5, 10.0)
    ph_layers = np.stack([ph - 0.15 * (bs < 0.9), ph, ph + 0.2 * (carb_left > 0) + 0.1])
    ph_layers = np.clip(ph_layers, 3.5, 10.0)

    # -- nitrogen -------------------------------------------------------------------------------
    cn_litter = 25.0 + 25.0 * tree
    n_org = (eq["dpm"] + eq["rpm"]) / cn_litter + eq["bio"] / CN_BIO + hum / CN_HUM + iom / CN_IOM
    f_nit = np.clip(m_rate, 0.0, 3.0) * (1.0 - wl)
    gross_min = (
        (K_DPM * eq["dpm"] + K_RPM * eq["rpm"]) * m_rate / cn_litter
        + K_BIO * m_rate * eq["bio"] / CN_BIO
        + K_HUM * m_rate * hum / CN_HUM
    )
    e = rothc_humification(100.0 * clay)
    immob = e * c_input / (1.0 - e) * (0.46 / CN_BIO + 0.54 / CN_HUM)
    net_min = gross_min - immob  # may be negative (immobilisation)
    fix = prm.n_fixation_max_kg_m2_yr * prod
    supply = np.maximum(net_min + fix + prm.n_deposition_kg_m2_yr, 0.0)
    k_up = prm.k_uptake * np.maximum(veg, 0.05)
    k_ni = prm.k_nitrify * f_nit
    nh4 = supply / (k_ni + k_up + 1e-9)
    oxic = np.clip(1.0 - wl * 1.2, 0.0, 1.0)
    k_den = prm.k_denit * (1.0 - oxic)
    k_le = q / theta_d
    no3 = k_ni * nh4 / (k_up + k_den + k_le + 1e-9)
    n_leach = k_le * no3
    n_denit = k_den * no3

    # -- oxygen and rooting ---------------------------------------------------------------------
    por = np.clip(np.nan_to_num(inp.porosity_mineral, nan=0.45), 0.2, 0.9)
    root_d = np.maximum(np.minimum(np.maximum(depth, 0.01), 1.0), 0.01)
    theta_m = pwp[None] + np.nan_to_num(inp.storage_month) / (
        1000.0 * root_d[None] * (1.0 - coarse[None]) + 1e-9
    )
    wfps = np.clip(theta_m / por[None], 0.0, 1.0)
    oxic_m = 1.0 - 1.0 / (1.0 + np.exp(-(wfps - 0.85) / 0.04))
    oxic_frac = np.clip(np.minimum(oxic_m.mean(axis=0), 1.0 - wl), 0.0, 1.0)

    # -- indices (0..1 each) ---------------------------------------------------------------------
    p_lab_top = xp[:, 1] * shares[0] + xp[:, 3] * 0.0
    s_n = 1.0 - np.exp(-np.maximum(supply, 0.0) / 0.006)
    s_p = 1.0 - np.exp(-(xp[:, 1] + 0.1 * xp[:, 2]) / 0.004)
    s_k = 1.0 - np.exp(-xk[:, 1] / 0.012)
    s_cec = np.clip(cec_ph7 / 15.0, 0.0, 1.0)
    nutrient = (s_n * s_p * s_k) ** (1.0 / 3.0) * (0.7 + 0.3 * s_cec)
    aet_ratio = np.clip(
        np.nan_to_num(inp.storage_month).mean(axis=0) / np.maximum(awc, 1.0), 0.0, 1.0
    )
    # water supply: storage capacity and how full the store is over the year
    water = np.sqrt(np.clip(awc / 150.0, 0.0, 1.0)) * np.sqrt(aet_ratio)
    soil_t = t12.mean(axis=0)
    rooting = (
        np.clip(depth / 1.0, 0.0, 1.0) * (1.0 - coarse) * np.clip((soil_t + 5.0) / 10.0, 0.0, 1.0)
    )
    slope_free = 1.0  # the slope enters through the erosion risk, not the fertility
    erod_k = 0.0293 * (0.65 - 2.0 * np.clip(clay, 0, 0.5)) + 0.04 * sand * (1.0 - sand)
    erod_k = np.clip(erod_k * (1.0 - np.clip(oc_top_frac * 10.0, 0.0, 0.5)), 0.005, 0.07)
    erosion_risk = np.clip(erosion / 1e-4, 0.0, 1.0) * (1.0 - 0.7 * veg) * slope_free

    out = {
        "soil_age_yr": age,
        "layer_thickness_m": thick,
        "bulk_density_kg_m3": bd_layers,
        "soc_layers_kg_m2": soc_layers,
        "soc_kg_m2": soc,
        "soc_pools": {"dpm": eq["dpm"], "rpm": eq["rpm"], "bio": eq["bio"], "hum": hum, "iom": iom},
        "litter_input_kg_c_m2_yr": c_input,
        "decomposition_modifier": m_rate,
        "n_org_kg_m2": n_org,
        "nh4_kg_m2": nh4,
        "no3_kg_m2": no3,
        "n_net_mineralisation_kg_m2_yr": net_min,
        "n_fixation_kg_m2_yr": fix,
        "n_leaching_kg_m2_yr": n_leach,
        "n_denitrification_kg_m2_yr": n_denit,
        "p_primary_kg_m2": xp[:, 0],
        "p_labile_kg_m2": xp[:, 1],
        "p_sorbed_kg_m2": xp[:, 2],
        "p_organic_kg_m2": xp[:, 3],
        "p_occluded_kg_m2": xp[:, 4],
        "p_labile_top_kg_m2": p_lab_top,
        "k_primary_kg_m2": xk[:, 0],
        "k_exchangeable_kg_m2": xk[:, 1],
        "k_solution_kg_m2": xk[:, 2],
        "bc_exchangeable_molc_m2": exch_bc_molc,
        "carbonate_kg_m2": carb_left,
        "cec_cmolc_kg": cec_ph7,
        "cec_molc_m2": cec_molc,
        "base_saturation": bs,
        "ph": ph,
        "ph_layers": ph_layers,
        "acid_equivalent_molc_m2": acid_eq,
        "buffer_molc_m2_per_ph": buffer,
        "salt_kg_m2": salt,
        "ec_ds_m": ec,
        "esp_pct": esp,
        "oxic_fraction": oxic_frac,
        "weathering_degree": oxide,
        "idx_nutrient": nutrient,
        "idx_water": water,
        "idx_rooting": rooting,
        "idx_oxygen": oxic_frac,
        "erodibility_k": erod_k,
        "erosion_risk": erosion_risk,
    }
    for k_, v in list(out.items()):
        if isinstance(v, np.ndarray):
            if v.ndim == 1:
                out[k_] = np.where(present, v, np.nan)
            else:
                out[k_] = np.where(present[None], v, np.nan)
        elif isinstance(v, dict):
            out[k_] = {kk: np.where(present, vv, np.nan) for kk, vv in v.items()}
    # element ledgers (over the soil age): initial + added = final + exported
    budget = {}
    for name, x_fin, x_ini, exp_, add_ in (
        ("P", xp.sum(axis=1), p_tot0, p_exp, p_add),
        ("K", xk.sum(axis=1), k_tot0, k_exp, k_add),
        ("BC", xb.sum(axis=1), bc_tot0, bc_exp, bc_add),
    ):
        res = np.abs(x_ini + add_ - x_fin - exp_)[present]
        scale = np.maximum(np.abs(x_ini + add_), 1e-12)[present]
        budget[name] = {
            "initial_kg": float((x_ini)[present].sum()),
            "added_kg": float(add_[present].sum()),
            "final_kg": float(x_fin[present].sum()),
            "exported_kg": float(exp_[present].sum()),
            "max_relative_residual": float((res / scale).max()) if res.size else 0.0,
        }
    out["budget"] = budget
    out["soil_present"] = present
    return out


# -- suitability -----------------------------------------------------------------------------------
#: Requirement profiles [A]: pH optimum range and tolerance, salinity threshold a (dS/m) and
#: slope b (fraction per dS/m) of Maas & Hoffman (1977) [I], anoxia tolerance (0 none, 1 full),
#: minimum rooting depth (m), weights of the factors (nutrient, water, ph, oxygen, rooting,
#: salinity).
PROFILES = {
    "native_generic": dict(ph=(5.0, 7.8), ph_tol=1.5, ec_a=4.0, ec_b=0.06, anoxia=0.3, root=0.4),
    "forest": dict(ph=(4.5, 7.0), ph_tol=1.5, ec_a=2.0, ec_b=0.08, anoxia=0.2, root=0.8),
    "grassland": dict(ph=(5.5, 8.2), ph_tol=1.5, ec_a=4.0, ec_b=0.06, anoxia=0.3, root=0.3),
    "wetland_plants": dict(ph=(4.0, 8.0), ph_tol=1.5, ec_a=3.0, ec_b=0.05, anoxia=1.0, root=0.2),
    "halophyte": dict(ph=(6.5, 9.5), ph_tol=1.5, ec_a=15.0, ec_b=0.02, anoxia=0.5, root=0.3),
    "crop_wheat": dict(ph=(5.8, 8.0), ph_tol=1.0, ec_a=6.0, ec_b=0.071, anoxia=0.1, root=0.6),
    "crop_maize": dict(ph=(5.8, 7.5), ph_tol=1.0, ec_a=1.7, ec_b=0.12, anoxia=0.1, root=0.6),
    "crop_rice_paddy": dict(ph=(5.0, 7.5), ph_tol=1.2, ec_a=3.0, ec_b=0.12, anoxia=1.0, root=0.3),
    "crop_cassava": dict(ph=(4.5, 7.0), ph_tol=1.2, ec_a=1.5, ec_b=0.10, anoxia=0.05, root=0.5),
    "crop_barley": dict(ph=(6.0, 8.5), ph_tol=1.0, ec_a=8.0, ec_b=0.05, anoxia=0.1, root=0.5),
}
FACTORS = ("nutrient", "water", "ph", "oxygen", "rooting", "salinity")
#: nutrient demand of a profile relative to the native vegetation (crops need more)
DEMAND = {
    "crop_wheat": 1.3,
    "crop_maize": 1.5,
    "crop_rice_paddy": 1.3,
    "crop_cassava": 0.8,
    "crop_barley": 1.2,
    "wetland_plants": 0.7,
    "halophyte": 0.7,
}


def ph_score(ph, lo, hi, tol):
    ph = np.asarray(ph, dtype=np.float64)
    below = np.clip(1.0 - (lo - ph) / tol, 0.0, 1.0)
    above = np.clip(1.0 - (ph - hi) / tol, 0.0, 1.0)
    return np.where(ph < lo, below, np.where(ph > hi, above, 1.0))


def salinity_score(ec, a, b):
    """Maas-Hoffman relative yield ``1 - b (EC - a)`` above the threshold ``a`` [I]."""
    return np.clip(1.0 - b * np.maximum(np.asarray(ec, dtype=np.float64) - a, 0.0), 0.0, 1.0)


def factor_scores(state: dict, profile: str) -> np.ndarray:
    """(6, n) factor scores of one profile (FACTORS order)."""
    p = PROFILES[profile]
    demand = DEMAND.get(profile, 1.0)
    nut = np.clip(state["idx_nutrient"] / demand, 0.0, 1.0)
    water = np.clip(state["idx_water"], 0.0, 1.0)
    ph = ph_score(state["ph"], *p["ph"], p["ph_tol"])
    al_tox = (state["ph"] < 5.0) & (state["base_saturation"] < 0.3)
    if profile not in ("crop_cassava", "forest", "wetland_plants"):
        ph = np.where(al_tox, ph * 0.6, ph)
    oxy = 1.0 - (1.0 - p["anoxia"]) * (1.0 - np.clip(state["idx_oxygen"], 0.0, 1.0))
    root = np.clip(
        np.nan_to_num(state["idx_rooting"]) / max(p["root"], 1e-6) * 0.5
        + 0.5 * np.clip(state["idx_rooting"], 0, 1),
        0.0,
        1.0,
    )
    sal = salinity_score(state["ec_ds_m"], p["ec_a"], p["ec_b"])
    sod = np.clip(1.0 - np.maximum(state["esp_pct"] - 15.0, 0.0) / 40.0, 0.0, 1.0)
    if profile != "halophyte":
        sal = sal * sod
    return np.stack([nut, water, ph, oxy, root, sal])


def suitability(state: dict, profile: str, alpha: float = 0.5, weights=None):
    """``S = min(s)^alpha (prod s^w)^(1-alpha)`` and the index of the limiting factor."""
    s = np.clip(np.nan_to_num(factor_scores(state, profile), nan=0.0), 1e-6, 1.0)
    w = np.full(len(FACTORS), 1.0 / len(FACTORS)) if weights is None else np.asarray(weights)
    w = w / w.sum()
    geo = np.exp((w[:, None] * np.log(s)).sum(axis=0))
    smin = s.min(axis=0)
    val = smin**alpha * geo ** (1.0 - alpha)
    lim = np.argmin(s, axis=0).astype(np.uint8)
    present = np.isfinite(state["ph"])
    return np.where(present, val, np.nan), np.where(present, lim, 255).astype(np.uint8), s


# -- monthly dynamics with an exact ledger (G3) ----------------------------------------------------
POOLS = (
    "c_dpm",
    "c_rpm",
    "c_bio",
    "c_hum",
    "c_iom",
    "n_org",
    "n_nh4",
    "n_no3",
    "p_primary",
    "p_labile",
    "p_sorbed",
    "p_organic",
    "p_occluded",
    "k_primary",
    "k_exchangeable",
    "k_solution",
    "salt",
)


def state_from_static(st: dict) -> dict:
    """Monthly state initialised from the static solution (flat arrays)."""
    sp = st["soc_pools"]
    return {
        "c_dpm": sp["dpm"],
        "c_rpm": sp["rpm"],
        "c_bio": sp["bio"],
        "c_hum": sp["hum"],
        "c_iom": sp["iom"],
        "n_org": st["n_org_kg_m2"],
        "n_nh4": st["nh4_kg_m2"],
        "n_no3": st["no3_kg_m2"],
        "p_primary": st["p_primary_kg_m2"],
        "p_labile": st["p_labile_kg_m2"],
        "p_sorbed": st["p_sorbed_kg_m2"],
        "p_organic": st["p_organic_kg_m2"],
        "p_occluded": st["p_occluded_kg_m2"],
        "k_primary": st["k_primary_kg_m2"],
        "k_exchangeable": st["k_exchangeable_kg_m2"],
        "k_solution": st["k_solution_kg_m2"],
        "salt": st["salt_kg_m2"],
    }


def _take(stock, flux):
    """A flux limited to the stock (kg/m2 over the step): never a negative mass."""
    return np.minimum(np.maximum(flux, 0.0), np.maximum(stock, 0.0))


def advance_soil_month(state: dict, month: dict, dt_days: float = 30.4375) -> tuple[dict, dict]:
    """One explicit month of C / N / P / K / salt with every flux bounded by its stock.

    ``month`` holds flat arrays: ``t_c``, ``moisture_mod`` (RothC b), ``cover_mod``,
    ``anoxia`` (0..1 share of the month saturated), ``litter_c`` (kg C/m2 in the month),
    ``litter_cn``, ``dpm_share``, ``clay_pct``, ``drain_mm``, ``theta_d_m`` (water held,
    m), ``uptake_n``, ``uptake_p``, ``uptake_k`` (kg/m2 demand of the month), ``n_fix``,
    ``n_dep``, ``salt_in``, ``weather_p``, ``weather_k`` (1/yr rate constants),
    ``k_sorb``, ``k_desorb``. Returns the new state and the flux report with the
    ledger per element (``residual`` = in - out - change, ~1e-15).
    """
    dt = dt_days / 365.25
    s = {k: np.array(v, dtype=np.float64) for k, v in state.items()}
    m = rothc_temperature(month["t_c"]) * month["moisture_mod"] * month["cover_mod"]
    m = m * (1.0 - 0.85 * month["anoxia"])
    before = _totals(s)
    rep = {}
    # carbon: decay of every pool, humification share e back to BIO / HUM
    e = rothc_humification(month["clay_pct"])
    dec = {}
    for pool, k in (("c_dpm", K_DPM), ("c_rpm", K_RPM), ("c_bio", K_BIO), ("c_hum", K_HUM)):
        rem = s[pool] * np.exp(-k * m * dt)
        dec[pool] = s[pool] - rem
        s[pool] = rem
    total_dec = sum(dec.values())
    s["c_bio"] += 0.46 * e * total_dec
    s["c_hum"] += 0.54 * e * total_dec
    co2 = (1.0 - e) * total_dec
    s["c_dpm"] += month["litter_c"] * month["dpm_share"]
    s["c_rpm"] += month["litter_c"] * (1.0 - month["dpm_share"])
    rep["c_in"], rep["c_co2"] = month["litter_c"], co2
    # nitrogen: litter N in, gross mineralisation of decomposed N, immobilisation into new
    # BIO / HUM at their C:N (taken from mineral N, never below zero)
    n_litter = month["litter_c"] / month["litter_cn"]
    cn_pool = {
        "c_dpm": month["litter_cn"],
        "c_rpm": month["litter_cn"],
        "c_bio": CN_BIO,
        "c_hum": CN_HUM,
    }
    n_dec = sum(dec[p] / cn_pool[p] for p in dec)
    n_dec = _take(s["n_org"], n_dec)
    s["n_org"] += n_litter - n_dec
    s["n_nh4"] += n_dec
    n_need = e * total_dec * (0.46 / CN_BIO + 0.54 / CN_HUM)
    from_nh4 = _take(s["n_nh4"], n_need)
    s["n_nh4"] -= from_nh4
    from_no3 = _take(s["n_no3"], n_need - from_nh4)
    s["n_no3"] -= from_no3
    s["n_org"] += from_nh4 + from_no3
    s["n_nh4"] += month["n_fix"] + month["n_dep"]
    nit = _take(
        s["n_nh4"],
        20.0 * rothc_temperature(month["t_c"]) / 3.0 * (1.0 - month["anoxia"]) * s["n_nh4"] * dt,
    )
    s["n_nh4"] -= nit
    s["n_no3"] += nit
    up_nh4 = _take(s["n_nh4"], 0.5 * month["uptake_n"])
    s["n_nh4"] -= up_nh4
    up_no3 = _take(s["n_no3"], month["uptake_n"] - up_nh4)
    s["n_no3"] -= up_no3
    den = _take(s["n_no3"], 8.0 * month["anoxia"] * s["n_no3"] * dt)
    s["n_no3"] -= den
    conc = s["n_no3"] / np.maximum(month["theta_d_m"], 1e-6)  # kg N per m3 of soil water
    leach = _take(s["n_no3"], month["drain_mm"] / 1000.0 * conc)
    s["n_no3"] -= leach
    rep.update(
        n_in=n_litter + month["n_fix"] + month["n_dep"],
        n_uptake=up_nh4 + up_no3,
        n_denit=den,
        n_leach=leach,
    )
    # phosphorus
    w = _take(s["p_primary"], month["weather_p"] * s["p_primary"] * dt)
    s["p_primary"] -= w
    s["p_labile"] += w
    ads = _take(s["p_labile"], month["k_sorb"] * s["p_labile"] * dt)
    des = _take(s["p_sorbed"], month["k_desorb"] * s["p_sorbed"] * dt)
    s["p_labile"] += des - ads
    s["p_sorbed"] += ads - des
    p_min = _take(s["p_organic"], 0.05 * m * s["p_organic"] * dt)
    s["p_organic"] -= p_min
    s["p_labile"] += p_min
    p_up = _take(s["p_labile"], month["uptake_p"])
    s["p_labile"] -= p_up
    p_ret = month.get("litter_p", np.zeros_like(p_up))
    s["p_organic"] += p_ret
    rep.update(p_in=p_ret, p_uptake=p_up)
    # potassium
    wk = _take(s["k_primary"], month["weather_k"] * s["k_primary"] * dt)
    s["k_primary"] -= wk
    s["k_solution"] += wk
    k_up = _take(s["k_solution"] + s["k_exchangeable"], month["uptake_k"])
    from_sol = np.minimum(k_up, s["k_solution"])
    s["k_solution"] -= from_sol
    s["k_exchangeable"] -= k_up - from_sol
    k_conc = s["k_solution"] / np.maximum(month["theta_d_m"], 1e-6)
    k_leach = _take(s["k_solution"], month["drain_mm"] / 1000.0 * k_conc)
    s["k_solution"] -= k_leach
    k_ret = month.get("litter_k", np.zeros_like(k_up))
    s["k_solution"] += k_ret
    rep.update(k_in=k_ret, k_uptake=k_up, k_leach=k_leach)
    # salt: inputs stay in the soil whatever the evaporation; only drainage removes salt
    s["salt"] += month["salt_in"]
    salt_conc = s["salt"] / np.maximum(month["theta_d_m"], 1e-6)
    salt_out = _take(s["salt"], month["drain_mm"] / 1000.0 * salt_conc)
    s["salt"] -= salt_out
    rep.update(salt_in=month["salt_in"], salt_leach=salt_out)
    after = _totals(s)
    ins = {
        "C": rep["c_in"],
        "N": rep["n_in"],
        "P": rep["p_in"],
        "K": rep["k_in"],
        "salt": rep["salt_in"],
    }
    outs = {
        "C": rep["c_co2"],
        "N": rep["n_uptake"] + rep["n_denit"] + rep["n_leach"],
        "P": rep["p_uptake"],
        "K": rep["k_uptake"] + rep["k_leach"],
        "salt": rep["salt_leach"],
    }
    rep["ledger"] = {
        el: {
            "in": ins[el],
            "out": outs[el],
            "change": after[el] - before[el],
            "residual": ins[el] - outs[el] - (after[el] - before[el]),
        }
        for el in ins
    }
    return s, rep


def _totals(s: dict) -> dict:
    return {
        "C": s["c_dpm"] + s["c_rpm"] + s["c_bio"] + s["c_hum"] + s["c_iom"],
        "N": s["n_org"] + s["n_nh4"] + s["n_no3"],
        "P": s["p_primary"] + s["p_labile"] + s["p_sorbed"] + s["p_organic"] + s["p_occluded"],
        "K": s["k_primary"] + s["k_exchangeable"] + s["k_solution"],
        "salt": s["salt"].copy(),
    }


def transfer_pool(state: dict, src: str, dst: str, amount) -> dict:
    """Move ``amount`` (kg/m2, limited to the source) between pools of one element."""
    s = dict(state)
    a = _take(s[src], amount)
    s[src] = s[src] - a
    s[dst] = s[dst] + a
    return s
