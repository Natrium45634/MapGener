"""v5.1: volcanism - provinces, reservoirs, eruptions, products, hazards, ecology (TZ 5.1)."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from hexworld import generate
from hexworld.config import Config, ConfigError, VolcanismParams
from hexworld.io import load_project, save_project
from hexworld.volcanism import environment as venv
from hexworld.volcanism import hazards, lahars, lava, scheduler, styles, tephra
from hexworld.volcanism.magma import (
    INTRUSIVE_SHARE,
    PROFILES,
    Reservoir,
    log10_viscosity,
    magma_for,
)
from hexworld.volcanism.systems import VolcanoSystem
from hexworld.volcanism.units import (
    L_FUSION_ICE,
    plume_height_m,
    seawater_pressure_pa,
    stable_hash,
    stable_uniform,
    tropopause_m,
    vei_from_bulk,
)


def _system(profile="arc_continental", edifice="stratocone", env="land", status="active_system",
            supply=300.0, crit=5e12, source="subduction_arc", depth=0.0, vid="V-test"):  # fmt: skip
    s = VolcanoSystem(
        volcano_id=vid, province_id="P-test", source_type=source, edifice_type=edifice,
        environment=env, magmatic_status=status, activity_phase="quiet", x_km=0.0, y_km=0.0,
        vents=[{"vent_id": f"{vid}-c", "x_km": 0.0, "y_km": 0.0, "kind": "central"}],
        summit_m=2500.0, radius_km=8.0, reactivation_probability=0.5, depth_m=depth,
        magma=magma_for(profile, vid), in_window=True,
    )  # fmt: skip
    s.reservoir = Reservoir.steady(supply if status != "extinct_system" else 0.0, crit, 0.6,
                                   INTRUSIVE_SHARE.get(source, 0.8))  # fmt: skip
    return s


# -- G0: configuration, registry, versions -------------------------------------------------------
def test_g0_default_params_are_valid():
    Config(volcanism=VolcanismParams())


@pytest.mark.parametrize(
    "bad",
    [
        {"world_profile": "lava_world"},
        {"physics_quality": "ultra"},
        {"event_window_years": 10.0},
        {"max_explicit_events": -1},
        {"tephra_bins": 7},
        {"lava_cell_m": 5.0},
        {"eruption_rate_scale": -1.0},
    ],
)
def test_g0_bad_params_are_rejected(bad):
    with pytest.raises(ConfigError):
        Config(volcanism=VolcanismParams(**bad))


def test_g0_config_round_trip_keeps_volcanism():
    c = Config(volcanism=VolcanismParams(eruption_rate_scale=2.0, world_profile="icy_world"))
    d = Config.from_dict(c.to_dict())
    assert d.volcanism == c.volcanism


def test_g0_settings_schema_covers_volcanism():
    from hexworld.settings_schema import PARAMS

    paths = {p.path for p in PARAMS}
    for name in VolcanismParams.__dataclass_fields__:
        assert f"volcanism.{name}" in paths


def test_g0_versions_and_stage_registered():
    from hexworld.progress import STAGES
    from hexworld.version import VOLCANISM_ALGORITHM_VERSION

    assert VOLCANISM_ALGORITHM_VERSION.startswith("hexworld-volcanism-")
    keys = [k for k, *_ in STAGES]
    assert keys.index("v51_volcanism") < keys.index("v50_biomes")


def test_g0_new_habitat_modifiers_are_appended():
    from hexworld.biomes.registry import MODIFIERS

    assert MODIFIERS[:3] == ("river_oasis", "saline_soil", "volcanic_ash_young")
    for m in ("geothermal_habitat", "volcanic_degassing_stress", "hydrothermal_vent_field"):
        assert m in MODIFIERS


# -- G1: units and magma ----------------------------------------------------------------------
def test_g1_stable_uniform_is_deterministic_and_open():
    a = [stable_uniform("s", i) for i in range(500)]
    assert a == [stable_uniform("s", i) for i in range(500)]
    assert 0.0 < min(a) and max(a) < 1.0
    assert abs(np.mean(a) - 0.5) < 0.05
    assert stable_hash("a", 1) != stable_hash("a", 2)


def test_g1_plume_height_matches_mastin():
    assert plume_height_m(1.0) == pytest.approx(2000.0)
    # 1e4 m^3/s DRE (Pinatubo-class) -> ~18 km
    assert 15_000 < plume_height_m(1e4) < 22_000
    assert plume_height_m(0.0) == 0.0


@pytest.mark.parametrize("bulk,vei", [(1e3, 0), (5e5, 1), (5e6, 2), (5e7, 3), (5e8, 4),
                                       (5e9, 5), (5e10, 6), (5e11, 7), (5e12, 8)])  # fmt: skip
def test_g1_vei_from_bulk_volume(bulk, vei):
    assert vei_from_bulk(bulk) == vei


def test_g1_seawater_pressure_and_tropopause():
    assert seawater_pressure_pa(0) == pytest.approx(101325.0)
    assert seawater_pressure_pa(1000) == pytest.approx(101325 + 1025 * 9.81 * 1000)
    assert tropopause_m(0) > 15000 > tropopause_m(45) > tropopause_m(85)


def test_g1_viscosity_trends():
    base = log10_viscosity(50, 1150, 0.1, 1.0)
    assert log10_viscosity(70, 1150, 0.1, 1.0) > base  # silica
    assert log10_viscosity(50, 900, 0.1, 1.0) > base  # cooler
    assert log10_viscosity(50, 1150, 0.4, 1.0) > base  # crystals
    assert log10_viscosity(50, 1150, 0.1, 4.0) < base  # water


def test_g1_profiles_have_typical_viscosities():
    eta = {k: magma_for(k, "x").log10_viscosity_pa_s for k in PROFILES}
    assert 0.5 < eta["mor_basalt"] < 3.0
    assert 4.0 < eta["arc_continental"] < 7.0
    assert eta["rhyolite_caldera"] > 8.0
    assert magma_for("mor_basalt", "x").magma_class == "basaltic"
    assert magma_for("rhyolite_caldera", "x").magma_class == "rhyolitic"


def test_g1_magma_spread_is_deterministic_and_small():
    a, b = magma_for("arc_oceanic", "v1"), magma_for("arc_oceanic", "v1")
    assert a == b
    c = magma_for("arc_oceanic", "v2")
    assert abs(c.silica_wt_pct - PROFILES["arc_oceanic"][0]) <= 2.0
    assert abs(c.temperature_c - PROFILES["arc_oceanic"][1]) <= 30.0


def test_g1_reservoir_budget_closes_over_a_history():
    r = Reservoir.steady(500.0, 5e12, 0.5, 0.8)
    for k in range(200):
        r.advance(10.0 + k % 7)
        r.withdraw(1e9 * (k % 5))
    rel = abs(r.budget_residual_kg()) / (r.supplied_kg + r.initial_kg)
    assert rel < 1e-12


def test_g1_reservoir_without_supply_only_freezes():
    r = Reservoir.steady(0.0, 5e12, 0.8)
    m0 = r.melt_kg
    r.advance(5000.0)
    assert r.melt_kg == pytest.approx(m0 * math.exp(-1.0), rel=1e-9)
    assert r.supplied_kg == 0.0 and r.erupted_kg == 0.0


def test_g1_no_eruption_without_melt():
    r = Reservoir.steady(0.0, 5e12, 0.0)
    assert r.withdraw(1e12) == 0.0
    assert r.melt_kg == 0.0


def test_g1_intrusive_share_keeps_most_supply_in_the_crust():
    r = Reservoir.steady(1000.0, 5e12, 0.0, INTRUSIVE_SHARE["mid_ocean_ridge"])
    r.advance(100.0)
    assert r.intruded_kg == pytest.approx(0.9 * r.supplied_kg)
    assert r.pressure_proxy >= 0.0


# -- G2: styles and the mass partition -------------------------------------------------------
def test_g2_deep_submarine_is_effusive_without_column():
    s = _system("mor_basalt", "fissure_field", "deep_submarine", source="mid_ocean_ridge",
                depth=2500.0)  # fmt: skip
    phases, _ = styles.choose_style(s, 1e7, "k")
    assert phases == ["submarine_effusive"]
    p = styles.partition(s, 1e7, phases, 10.0)
    assert p["plume_height_m"] == 0.0 and p["vei"] == 0


def test_g2_subglacial_and_shallow_water_make_phreatomagmatic():
    s = _system("arc_oceanic", "stratocone", "subglacial")
    assert styles.choose_style(s, 1e7, "k")[0][0] == "phreatomagmatic_surtseyan"
    s2 = _system("ocean_island_basalt", "submarine_seamount", "shallow_submarine", depth=80.0,
                 source="oceanic_hotspot")  # fmt: skip
    assert styles.choose_style(s2, 1e7, "k")[0][0] == "phreatomagmatic_surtseyan"


def test_g2_basalt_is_effusive_and_rhyolite_is_explosive():
    b = _system("mor_basalt", "shield", "land", source="oceanic_hotspot")
    assert styles.choose_style(b, 1e8, "k")[0][0] in ("hawaiian_effusive", "strombolian")
    r = _system("rhyolite_caldera", "caldera_complex", "land", source="continental_hotspot")
    assert styles.choose_style(r, 5e9, "k")[0][0] in ("plinian", "subplinian")


def test_g2_ultraplinian_needs_eligibility_and_size():
    r = _system("rhyolite_caldera", "caldera_complex", "land", source="continental_hotspot")
    assert "ultraplinian" not in styles.choose_style(r, 5e11, "k")[0]
    r.supereruption_eligible = True
    assert styles.choose_style(r, 5e11, "k")[0][0] == "ultraplinian"


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_g2_partition_closes_the_mass(profile):
    s = _system(profile)
    for dre in (1e6, 1e8, 1e10):
        phases, _ = styles.choose_style(s, dre, f"k{dre}")
        p = styles.partition(s, dre, phases, 30.0)
        total = p["lava_mass_kg"] + p["tephra_mass_kg"] + p["gas_mass_kg"]
        assert total == pytest.approx(p["magma_mass_kg"], rel=1e-12)
        assert p["tephra_bulk_volume_m3"] >= p["tephra_dre_m3"]  # bulk vs DRE kept apart


def test_g2_stratospheric_so2_only_above_the_tropopause():
    s = _system("arc_continental")
    p = styles.partition(s, 1e9, ["plinian"], 10.0)
    assert p["plume_height_m"] > tropopause_m(10.0)
    assert p["so2_stratospheric_kg"] == p["so2_mass_kg"] > 0
    q = styles.partition(s, 1e6, ["strombolian"], 10.0)
    assert q["so2_stratospheric_kg"] == 0.0


def test_g2_effusive_has_vei_zero_and_explosive_has_pdc_flag():
    s = _system("arc_continental")
    assert styles.partition(s, 1e7, ["hawaiian_effusive"], 30)["vei"] == 0
    assert styles.partition(s, 1e9, ["plinian"], 30)["pdc"]


def test_g2_style_choice_is_deterministic():
    s = _system("arc_oceanic")
    assert styles.choose_style(s, 3e7, "E1") == styles.choose_style(s, 3e7, "E1")


# -- G2: the event scheduler -----------------------------------------------------------------
def test_g2_schedule_is_deterministic_with_unique_ids():
    prm = VolcanismParams()
    a = scheduler.schedule(_system(), "seedA", prm)
    b = scheduler.schedule(_system(), "seedA", prm)
    assert [e.event_id for e in a] == [e.event_id for e in b]
    assert len({e.event_id for e in a}) == len(a) > 0
    assert all(0 <= e.years_bp <= prm.event_window_years for e in a)
    assert [e.years_bp for e in a] == sorted((e.years_bp for e in a), reverse=True)


def test_g2_other_seed_gives_other_history():
    prm = VolcanismParams()
    a = scheduler.schedule(_system(), "seedA", prm)
    b = scheduler.schedule(_system(), "seedB", prm)
    assert [e.event_id for e in a] != [e.event_id for e in b]


def test_g2_extinct_system_never_erupts():
    s = _system(status="extinct_system")
    assert scheduler.intensity(s, VolcanismParams()) == 0.0
    assert scheduler.schedule(s, "x", VolcanismParams()) == []


def test_g2_rate_scale_increases_eruptions():
    lo = scheduler.schedule(_system(), "s", VolcanismParams(eruption_rate_scale=0.3))
    hi = scheduler.schedule(_system(), "s", VolcanismParams(eruption_rate_scale=3.0))
    assert len(hi) > len(lo)


def test_g2_erupted_mass_never_exceeds_the_supply():
    s = _system()
    ev = scheduler.schedule(s, "s", VolcanismParams())
    r = s.reservoir
    erupted = sum(e.products["magma_mass_kg"] for e in ev)
    assert erupted == pytest.approx(r.erupted_kg, rel=1e-9)
    assert erupted <= r.initial_kg + r.supplied_kg
    assert abs(r.budget_residual_kg()) / (r.initial_kg + r.supplied_kg) < 1e-12


def test_g2_pressure_factor_bounds():
    assert scheduler.pressure_factor(0.0) == pytest.approx(0.05**2)
    assert scheduler.pressure_factor(1.0) == 1.0
    assert scheduler.pressure_factor(100.0) == 9.0


def test_g2_dormant_system_reactivates_with_its_probability():
    prm = VolcanismParams()
    n = 0
    for i in range(200):
        s = _system(status="dormant_system", vid=f"V-d{i}")
        s.reactivation_probability = 0.3
        n += bool(scheduler.schedule(s, "s", prm))
    assert 30 < n < 95  # 60 expected; reservoir limits can only lower it


def test_g2_explicit_selection_caps_deep_submarine():
    events = []
    for env in ("deep_submarine", "land"):
        for e in scheduler.schedule(_system(env=env, vid=f"V-{env}", supply=3000.0), "s",
                                    VolcanismParams()):  # fmt: skip
            events.append(e)
    scheduler.select_explicit(events, 20, set())
    chosen = [e for e in events if e.explicit]
    assert len(chosen) <= 20
    assert sum(e.environment == "deep_submarine" for e in chosen) <= 5


def test_g2_aggregates_conserve_the_totals():
    ev = scheduler.schedule(_system(), "s", VolcanismParams())
    scheduler.select_explicit(ev, 5, {"V-test"})
    agg = scheduler.aggregate(ev)
    tot = sum(e.products["volume_dre_m3"] for e in ev)
    expl = sum(e.products["volume_dre_m3"] for e in ev if e.explicit)
    assert agg["V-test"]["dre_m3"] + expl == pytest.approx(tot, rel=1e-12)


def test_g2_lava_lake_switch_is_respected():
    s = _system("mor_basalt", "shield", "land", supply=8000.0, source="oceanic_hotspot")
    scheduler.schedule(s, "s", VolcanismParams(lava_lakes=False))
    assert s.lava_lake_state == "none"


# -- G3: lava on the local D6 lattice -----------------------------------------------------------
def test_g3_lattice_neighbours_are_symmetric():
    lat = lava.Lattice(0.0, 0.0, 6, 400.0)
    for i in range(lat.size):
        for k in range(6):
            j = lat.nb[i, k]
            if j >= 0:
                assert lat.nb[j, (k + 3) % 6] == i
    assert lat.nb[lat.vent].min() >= 0


def test_g3_volume_is_conserved_on_flat_ground():
    lat = lava.Lattice(0.0, 0.0, 30, 400.0)
    z = np.zeros(lat.size)
    r = lava.simulate(z, lat, 5e7, 2.0, 50.0)
    assert abs(r["budget_residual_m3"]) / r["injected_m3"] < 1e-12
    assert r["placed_m3"] + r["escaped_m3"] == pytest.approx(5e7, rel=1e-12)


def test_g3_lava_runs_downhill():
    lat = lava.Lattice(0.0, 0.0, 30, 400.0)
    z = -50.0 * lat.x + 1000.0  # slope down to +x (z in m, x in km)
    r = lava.simulate(z, lat, 3e7, 2.0, 50.0)
    h = r["thickness_m"]
    assert (h * lat.x).sum() / h.sum() > 0.3


def test_g3_higher_yield_makes_a_smaller_flow():
    lat = lava.Lattice(0.0, 0.0, 30, 400.0)
    z = np.zeros(lat.size)
    a = (lava.simulate(z, lat, 3e7, 1.5, 50.0)["thickness_m"] > 0.05).sum()
    b = (lava.simulate(z, lat, 3e7, 12.0, 50.0)["thickness_m"] > 0.05).sum()
    assert a > b


def test_g3_cooling_limits_the_length():
    lat = lava.Lattice(0.0, 0.0, 60, 400.0)
    z = -30.0 * lat.x + 1000.0
    long_ = lava.simulate(z, lat, 5e7, 1.5, 30.0)["thickness_m"]
    short = lava.simulate(z, lat, 5e7, 1.5, 1.0)["thickness_m"]
    assert lat.dist_km[long_ > 0.05].max() > lat.dist_km[short > 0.05].max()


def test_g3_cooling_length_scales_with_effusion_and_water():
    assert lava.cooling_length_km(100.0, "land") == pytest.approx(16.0)
    assert lava.cooling_length_km(100.0, "deep_submarine") < lava.cooling_length_km(100, "land")


def test_g3_lava_leaving_the_lattice_is_counted():
    lat = lava.Lattice(0.0, 0.0, 6, 400.0)
    z = -200.0 * lat.x
    r = lava.simulate(z, lat, 5e7, 1.0, 100.0)
    assert r["escaped_m3"] > 0
    assert abs(r["budget_residual_m3"]) / r["injected_m3"] < 1e-12


# -- G4: tephra, PDC, lahars ------------------------------------------------------------------
def test_g4_bin_shares_sum_to_one_and_merge():
    for phases in (["plinian"], ["strombolian", "hawaiian_effusive"], ["unknown"]):
        for nb in (2, 3, 4):
            s = tephra.bin_shares(phases, nb)
            assert len(s) == nb and s.sum() == pytest.approx(1.0)


def test_g4_upper_wind_has_a_westerly_jet():
    lat = np.array([0.0, 40.0, -40.0, 85.0])
    u = tephra.upper_wind_ms(lat)
    assert u[0] < 0 and u[1] > 20 and u[2] > 20 and abs(u[3]) < 5


def test_g4_pdc_runout_grows_with_volume_and_drop():
    assert hazards.energy_cone_reach_km(1e9, 1000) > hazards.energy_cone_reach_km(1e6, 1000)
    assert hazards.energy_cone_reach_km(1e8, 2000) > hazards.energy_cone_reach_km(1e8, 500)


def test_g4_lahar_needs_water_and_sediment():
    assert lahars.lahar_volume_m3(0.0, 1e7) == 0.0
    assert lahars.lahar_volume_m3(1e7, 0.0) == 0.0
    v = lahars.lahar_volume_m3(1e6, 1e8)
    assert v == pytest.approx(1e6 / 0.45)  # water-limited, bulked to c = 0.55


def test_g4_melt_is_energy_limited_and_capped_by_ice():
    m = lahars.melt_water_kg(1e9, 1000.0, 1e20)
    e = lahars.HEAT_TO_ICE * 1e9 * (lahars.C_P_MAGMA * 1000.0 + lahars.L_CRYST)
    assert m == pytest.approx(e / L_FUSION_ICE)
    assert lahars.melt_water_kg(1e9, 1000.0, 10.0) == 10.0


# -- G5: environment, submarine, ice, climate, icy world ----------------------------------------
def test_g5_hydrothermal_kinds():
    ridge = _system("mor_basalt", "fissure_field", "deep_submarine", source="mid_ocean_ridge",
                    depth=2500.0)  # fmt: skip
    assert venv.hydrothermal_field(ridge)["kind"] == "black_smoker"
    land = _system()
    assert venv.hydrothermal_field(land)["kind"] == "hot_spring_fumarole"
    assert venv.hydrothermal_field(_system(status="extinct_system")) is None


def test_g5_emergence_of_a_shallow_seamount():
    s = _system("ocean_island_basalt", "submarine_seamount", "shallow_submarine", depth=100.0,
                source="oceanic_hotspot")  # fmt: skip
    assert venv.emergence(s, 1e6) is None
    assert venv.emergence(s, 1e10) is not None
    assert venv.emergence(_system(), 1e12) is None  # land is not a seamount


def test_g5_climate_pulse_scale_and_threshold():
    class E:
        event_id, years_bp = "E-x", 30.0
        products = {"so2_stratospheric_kg": 17e9}

    p = venv.climate_pulse(E)
    assert -1.0 < p["peak_cooling_k"] < -0.4  # Pinatubo ~ -0.5..-0.7 K
    assert p["applied_to_present_climate"] is False
    E.products = {"so2_stratospheric_kg": 1e7}
    assert venv.climate_pulse(E) is None


def test_g5_icy_world_overrides_the_magma():
    s = _system()
    venv.icy_world_overrides(s)
    assert s.magma.sulfur_wt_pct == 0.0 and s.magma.temperature_c < 0
    assert s.magma.profile.startswith("cryomagma")


def test_g5_degassing_stress_only_on_land():
    s = _system("mor_basalt", "shield", "land", source="oceanic_hotspot")
    s.activity_phase = "degassing"
    assert venv.degassing_stress(s) > 0
    s.environment = "deep_submarine"
    assert venv.degassing_stress(s) == 0.0
    f = venv.stress_fraction_of_hex(8.0, 14.4)
    assert 0.0 < f < 1.0


# -- G6-G8: the stage on a generated world -------------------------------------------------
@pytest.fixture(scope="module")
def vproject(hist_project):
    assert hist_project.volcanism is not None
    return hist_project


def test_g6_stage_runs_and_validates(vproject):
    v = vproject.volcanism
    assert v.validation["ok"], v.validation["errors"]
    s = v.summary
    assert s["systems"] > 0 and s["events"] > 0
    assert s["explicit_events"] <= s["explicit_limit"]


def test_g6_budgets_close(vproject):
    checks = vproject.volcanism.validation["checks"]
    for k, val in checks.items():
        assert val < 1e-9, k


def test_g6_layers_have_valid_ranges(vproject):
    L = vproject.volcanism.layers
    shape = vproject.layer("elevation_m").shape
    for name, a in L.items():
        assert a.shape == shape, name
    names = ("lava_fraction", "pdc_deposit_fraction", "lahar_fraction",
             "volcanic_barren_fraction", "volcanic_hazard_index",
             "volcanic_ash_stress")  # fmt: skip
    for name in names:
        a = np.nan_to_num(L[name])
        assert a.min() >= 0 and a.max() <= 1 + 1e-6, name


def test_g6_no_barren_or_hazard_at_sea(vproject):
    L = vproject.volcanism.layers
    ocean = vproject.hydro.layers["ocean_mask"]
    assert np.nan_to_num(L["volcanic_barren_fraction"])[ocean].max(initial=0) == 0
    assert np.nan_to_num(L["volcanic_hazard_index"])[ocean].max(initial=0) == 0


def test_g6_systems_come_from_geology(vproject):
    src = {s["source_type"] for s in vproject.volcanism.systems}
    assert src <= {"subduction_arc", "oceanic_hotspot", "continental_hotspot", "mid_ocean_ridge",
                   "divergent_rift", "sedimentary_fluid_vent"}  # fmt: skip
    for s in vproject.volcanism.systems:
        assert s["volcano_id"].startswith("V-")


def test_g6_event_ids_unique_and_explicit_have_results(vproject):
    ev = vproject.volcanism.events
    assert len({e["event_id"] for e in ev}) == len(ev)
    assert all(e["explicit"] for e in ev)


def test_g6_no_cryovolcanism_on_a_silicate_world(vproject):
    for e in vproject.volcanism.events:
        assert "cryovolcanic" not in e["quality_flags"]


def test_g6_biomes_consume_the_forcing(vproject):
    from hexworld.biomes.registry import INDEX

    b = vproject.biomes
    fr = b.fractions
    off = fr["offsets"]
    barren = np.nan_to_num(vproject.volcanism.layers["volcanic_barren_fraction"]).ravel()
    code = INDEX["barren_volcanic_recent"]
    for h in np.nonzero(barren > 0.01)[0][:20]:
        a, z = off[h], off[h + 1]
        share = fr["fraction"][a:z][fr["biome"][a:z] == code].sum()
        assert share > 0


def test_g6_biome_key_depends_on_volcanism(vproject):
    from hexworld.pipeline_v50 import upstream_keys

    assert upstream_keys(vproject)["volcanism"] == vproject.volcanism.stage_keys


def test_g7_rerun_is_deterministic(vproject):
    from hexworld.pipeline_v51 import run_volcanism

    again = run_volcanism(vproject.config, vproject)
    for k, a in vproject.volcanism.layers.items():
        assert np.array_equal(np.nan_to_num(a), np.nan_to_num(again["layers"][k])), k
    assert [e["event_id"] for e in again["events"]] == [
        e["event_id"] for e in vproject.volcanism.events
    ]


def test_g7_save_load_round_trip(tmp_path, vproject):
    q = load_project(save_project(vproject, tmp_path / "w"))
    assert q.volcanism is not None
    assert q.volcanism.content_hashes() == vproject.volcanism.content_hashes()
    assert q.volcanism.validation["ok"]
    assert q.biomes is not None  # v5.0 key includes v5.1 and still matches


def test_g7_reuse_and_invalidation(hist_config, vproject):
    from hexworld.pipeline_v51 import upstream_keys, volcanism_stage_keys

    r = generate(hist_config, reuse=vproject)
    assert "v51_volcanism" in r.stats["reused_stages"]
    up = upstream_keys(vproject)
    k1 = volcanism_stage_keys(hist_config, up)
    k2 = volcanism_stage_keys(
        replace(hist_config, volcanism=VolcanismParams(eruption_rate_scale=2.0)), up
    )
    assert k1 != k2


def test_g7_old_save_without_volcanism_loads_as_not_generated(tmp_path, vproject):
    old = replace(vproject, volcanism=None)
    q = load_project(save_project(old, tmp_path / "old"))
    assert q.volcanism is None


def test_g7_icy_world_profile(vproject):
    from hexworld.pipeline_v51 import run_volcanism

    cfg = replace(vproject.config, volcanism=VolcanismParams(world_profile="icy_world"))
    res = run_volcanism(cfg, vproject)
    assert all(e["products"]["so2_mass_kg"] == 0 for e in res["events"])
    assert all("cryovolcanic" in e["quality_flags"] for e in res["events"])
    assert not res["extras"]["climate_pulses"]


def test_g7_disabled_volcanism_keeps_biomes(hist_config):
    from hexworld.pipeline_v50 import run_biomes

    p = generate(replace(hist_config, volcanism=VolcanismParams(enabled=False)), through="v47")
    assert getattr(p, "volcanism", None) is None
    res = run_biomes(p.config, p.geometry, p)
    assert res["summary"]["volcanic_fringe"] == {"hexes": 0}


def test_g8_view_renders_every_mode(vproject):
    from hexworld.views import VIEWS

    v = VIEWS["eruptions"]
    assert v.availability(vproject)[0]
    for m in v.modes:
        rgb = v.render(vproject, m.id, v.default_overlays())
        assert rgb.shape[:2] == vproject.layer("elevation_m").shape
        assert v.legend(vproject, m.id)
    assert isinstance(v.vectors(vproject, "status", v.default_overlays()), list)


def test_g8_view_unavailable_without_volcanism(vproject):
    from hexworld.views import VIEWS

    assert not VIEWS["eruptions"].availability(replace(vproject, volcanism=None))[0]


def test_g8_hex_report_has_a_volcanism_section(vproject):
    from hexworld.hex_report import hex_report

    sy = [s for s in vproject.volcanism.systems if s["hex"] is not None]
    assert sy
    c, r = sy[0]["hex"]
    rep = hex_report(vproject, int(c), int(r))
    assert any(sec.key == "volcanism" for sec in rep.sections)


def test_g8_cli_prints_the_summary(tmp_path, vproject, capsys):
    from hexworld.cli import main

    path = save_project(vproject, tmp_path / "w")
    assert main(["volcanism", str(path), "--events", "3"]) == 0
    out = capsys.readouterr().out
    assert "systems" in out and "validation: True" in out


def test_g8_explicit_limit_scales_with_world_size():
    from hexworld.pipeline_v51 import explicit_limit

    assert explicit_limit(160, 10_000) == 34
    assert explicit_limit(160, 1_000_000) == 160
    assert explicit_limit(0, 10_000) == 0


def test_g8_recovery_is_slower_in_cold_and_dry_climates():
    from hexworld.pipeline_v51 import recovery_years

    t = recovery_years(np.array([26.0, 10.0, 26.0, -5.0]), np.array([2500.0, 900, 200, 300]))
    assert t[0] < t[1] < t[3] and t[0] < t[2]
    assert 100 <= t[0] <= 250
