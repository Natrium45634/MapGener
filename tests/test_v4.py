"""v4.0: climate, hydrology, landscape evolution (ТЗ 4.0, section 9)."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

from hexworld import (
    generate,
    generate_history,
    generate_v4,
    load_project,
    save_project,
    views,
)
from hexworld.api import derive_landmasses, generate_climate
from hexworld.cli import main
from hexworld.config import ClimateParams, Config, ConfigError, HistoryParams
from hexworld.geometry import Bounds, HexLayout
from hexworld.stages.climate import (
    Latitude,
    hex_climate,
    prevailing_wind,
    resolve_latitude,
    world_climate,
)
from hexworld.stages.hydrology import (
    SECONDS_PER_YEAR,
    hex_graph,
    hydrology,
    river_network,
    river_systems,
    routing_choice,
)
from hexworld.stages.landscape import evolve

V2_FIXTURE = Path(__file__).parent / "data" / "v2_project"
GRID = {"x0_km": 0.0, "y0_km": 0.0, "cell_km": 60.0}


def _latitude(center, ny, km_per_deg=111.2):
    return Latitude(center, 0.5 * ny * GRID["cell_km"], km_per_deg, False)


def _continent(ny=60, nx=120, coast_col=30, ridge=0.0):
    """Ocean in the west, a flat 200 m continent in the east, optional N-S ridge at col 60."""
    cols = np.arange(nx)[None, :].repeat(ny, axis=0).astype(float)
    z = np.where(cols < coast_col, -3000.0, 200.0)
    if ridge:
        z = z + np.where(cols >= coast_col, ridge * np.exp(-(((cols - 60.0) / 4.0) ** 2)), 0.0)
    return z


# -- 3. climate ------------------------------------------------------------------------
def test_rain_shadow_windward_wet_leeward_dry():
    """Mid-latitude westerlies: the west slope of a ridge is wet, the east side is dry."""
    z = _continent(ridge=3000.0)
    wc = world_climate(ClimateParams(), GRID, z, 0.0, _latitude(45.0, z.shape[0]))
    band = slice(22, 39)  # ~ 40-50 deg N: westerlies
    assert (wc.wind_u[band, 40:80] > 0).all()  # the wind blows from the west (+x east)
    windward = wc.precipitation[band, 53:59].mean()
    leeward = wc.precipitation[band, 64:72].mean()
    assert windward > 1.6 * leeward, (windward, leeward)
    assert (wc.precipitation >= 0).all() and np.isfinite(wc.precipitation).all()


def test_maritime_margin_is_wetter_and_milder_than_the_interior():
    z = _continent()
    wc = world_climate(ClimateParams(), GRID, z, 0.0, _latitude(45.0, z.shape[0]))
    band = slice(22, 39)
    coast = wc.precipitation[band, 31:36].mean()
    interior = wc.precipitation[band, 105:115].mean()
    assert coast > 1.5 * interior, (coast, interior)
    assert wc.seasonal_amplitude[band, 110].mean() > wc.seasonal_amplitude[band, 32].mean()
    assert wc.balance["ocean_mean_mm_yr"] == pytest.approx(1100.0, rel=1e-6)


def test_temperature_falls_with_height_at_the_lapse_rate():
    z = _continent()
    wc = world_climate(ClimateParams(), GRID, z, 0.0, _latitude(45.0, z.shape[0]))
    layout = HexLayout(30, 20, 60.0, 2400.0, 900.0)  # inside the continent
    low = hex_climate(ClimateParams(), wc, layout, np.full((20, 30), 500.0), 0.0)
    high = hex_climate(ClimateParams(), wc, layout, np.full((20, 30), 2500.0), 0.0)
    drop = low["temperature_mean_c"] - high["temperature_mean_c"]
    assert np.allclose(drop, 13.0, atol=1e-3)  # 6.5 C/km * 2 km
    assert (high["snow_fraction"] >= low["snow_fraction"]).all()
    for name, arr in high.items():
        assert np.isfinite(arr).all(), name
    assert (high["actual_evapotranspiration_mm_yr"] <= high["precipitation_mm_yr"] + 1e-3).all()


def test_equator_outside_the_map_and_southern_hemisphere():
    z = _continent(ny=40)
    south = world_climate(ClimateParams(), GRID, z, 0.0, _latitude(-55.0, 40))
    assert south.lat.max() < -40.0 and south.lat.min() > -70.0
    assert np.isfinite(south.t_sea_level).all() and (south.precipitation >= 0).all()
    # westerlies in both hemispheres; the surface flow of the Ferrel cell is poleward
    u_n, v_n = prevailing_wind(np.array([45.0]), 1.0)
    u_s, v_s = prevailing_wind(np.array([-45.0]), 1.0)
    assert u_n[0] > 0 and u_s[0] > 0 and v_n[0] < 0 < v_s[0]  # +y is south
    tropics = world_climate(ClimateParams(), GRID, z, 0.0, _latitude(10.0, 40))
    assert tropics.t_sea_level.mean() > south.t_sea_level.mean() + 15.0


def test_latitude_scale_is_compressed_only_when_the_area_would_cross_a_pole():
    small = resolve_latitude(ClimateParams(latitude_center_deg=30.0), Bounds(0, 0, 4000, 3000))
    assert small.km_per_deg == pytest.approx(111.2, rel=1e-3) and not small.compressed
    huge = resolve_latitude(ClimateParams(latitude_center_deg=30.0), Bounds(0, 0, 4e4, 3e4))
    assert huge.compressed and huge.at(0.0) <= 90.0 and huge.at(3e4) >= -90.0
    fixed = resolve_latitude(ClimateParams(km_per_degree=50.0), Bounds(0, 0, 4e4, 3e4))
    assert fixed.km_per_deg == 50.0 and not fixed.compressed


def test_climate_parameters_are_validated():
    with pytest.raises(ConfigError):
        Config(climate=ClimateParams(profile="martian"))
    with pytest.raises(ConfigError):
        Config.from_dict({"hydrology": {"routing": "d8"}})


# -- 4. hydrology ----------------------------------------------------------------------
def _plane(width=60, height=40, spacing=50.0, slope_per_hex=30.0):
    layout = HexLayout(width, height, spacing, 0.0, 0.0)
    cols = np.arange(width)[None, :].repeat(height, axis=0).astype(float)
    z = 20.0 + slope_per_hex * cols
    z[:, 0] = -100.0  # sea along the west edge
    return layout, z


def _run(layout, z, runoff=300.0, precip=800.0, evap=1000.0, choice=None, loss=0.0):
    n = z.size
    zz = z.ravel()
    return hydrology(
        zz,
        zz <= 0.0,
        hex_graph(layout),
        layout.hex_area_km2 * 1e6,
        np.full(n, runoff),
        np.full(n, precip),
        np.full(n, evap),
        None,
        2.0,
        loss,
        choice,
    )


def _assert_flow_graph_is_sound(hyd):
    r = hyd["routing"]
    rc = hyd["receiver"]
    inner = rc != np.arange(rc.size)
    assert (r.rank[rc[inner]] < r.rank[inner]).all()  # acyclic: receivers come first
    assert (r.routing_m[rc] <= r.routing_m + 1e-9).all()  # never uphill on the routing surface
    assert (hyd["discharge_m3_s"] >= 0).all() and np.isfinite(hyd["discharge_m3_s"]).all()


@pytest.mark.parametrize("mode", ["steepest", "stochastic"])
def test_inclined_plane_drains_to_the_sea_and_conserves_water(mode):
    layout, z = _plane()
    hyd = _run(layout, z, choice=routing_choice("Plane1", z.size, mode))
    _assert_flow_graph_is_sound(hyd)
    land = z.ravel() > 0
    supplied = (0.3 * layout.hex_area_km2 * 1e6 / SECONDS_PER_YEAR) * land.sum()
    out = hyd["routing"].outlet
    assert hyd["discharge_m3_s"][out].sum() == pytest.approx(supplied, rel=1e-9)
    assert not hyd["lakes"] and not hyd["endorheic_mask"].any()


def test_stochastic_routing_converges_where_steepest_descent_runs_in_parallel():
    layout, z = _plane()
    area = layout.hex_area_km2
    steep = _run(layout, z)
    stoch = _run(layout, z, choice=routing_choice("Plane1", z.size, "stochastic"))
    steep_max = steep["area_km2"].max() / area
    stoch_max = stoch["area_km2"].max() / area
    assert steep_max <= 60  # a straight line across the plane, nothing joins it
    assert stoch_max > 2.5 * steep_max, (steep_max, stoch_max)
    # the draw is fixed by the seed
    again = _run(layout, z, choice=routing_choice("Plane1", z.size, "stochastic"))
    assert np.array_equal(again["receiver"], stoch["receiver"])


def _bowl(depth=400.0):
    layout, z = _plane(width=40, height=30)
    cx, cy = layout.center(20, 15)
    x, y = layout.centers()
    d2 = ((x - cx) ** 2 + (y - cy) ** 2) / (4 * layout.spacing_km) ** 2
    return layout, z - depth * np.exp(-d2)


def test_wet_depression_fills_and_spills_through_its_sill():
    layout, z = _bowl()
    hyd = _run(layout, z, runoff=500.0, precip=900.0, evap=800.0)
    _assert_flow_graph_is_sound(hyd)
    assert len(hyd["lakes"]) == 1
    lake = hyd["lakes"][0]
    assert not lake["closed"] and lake["outflow_m3_s"] > 0
    pit = 15 * 40 + 20
    assert hyd["lake_id"][pit] == 0 and hyd["lake_depth_m"][pit] > 50.0
    assert not hyd["endorheic_mask"][pit]
    # the physical surface is not raised: the lake level lives in its own layer
    assert hyd["lake_level_m"][pit] > z.ravel()[pit]


def test_arid_depression_becomes_an_endorheic_basin():
    layout, z = _bowl()
    hyd = _run(layout, z, runoff=3.0, precip=150.0, evap=2200.0)
    _assert_flow_graph_is_sound(hyd)
    pit = 15 * 40 + 20
    assert hyd["endorheic_mask"][pit]
    closed = [lk for lk in hyd["lakes"] if lk["closed"]]
    assert closed or hyd["dry_basins"]  # a terminal lake shrunk to balance, or a dry pan
    for lk in closed:
        assert lk["outflow_m3_s"] == 0.0


def test_river_network_tributaries_strahler_and_mouths():
    layout, z = _plane()
    hyd = _run(layout, z, choice=routing_choice("Rivers3", z.size, "stochastic"))
    land = z.ravel() > 0
    slope = np.full(z.size, 0.6)
    river, strahler, segments, mouths = river_network(
        hyd, land, layout, 5.0, slope, 3 * layout.hex_area_km2
    )
    assert river.any() and strahler.max() >= 2
    rc = hyd["receiver"]
    # Strahler: order rises by one only where two streams of the same top order meet
    for c in np.nonzero(river)[0]:
        donors = [d for d in np.nonzero((rc == c) & river)[0] if d != c]
        orders = sorted((int(strahler[d]) for d in donors), reverse=True)
        if not orders:
            assert strahler[c] == 1
        elif len(orders) > 1 and orders[0] == orders[1]:
            assert strahler[c] == orders[0] + 1
        else:
            assert strahler[c] == orders[0]
    by_id = {s["id"]: s for s in segments}
    for s in segments:  # every segment leads to a mouth through a finite chain
        seen = 0
        while s["next_segment"] >= 0:
            s = by_id[s["next_segment"]]
            seen += 1
            assert seen <= len(segments)
        assert s["mouth"] in ("sea", "window_edge", "lake", "endorheic")
        assert len(s["x_km"]) == len(s["y_km"]) >= 2
    systems = river_systems(segments)
    assert systems and all(r["main_stem_length_km"] > 0 for r in systems)
    # generalisation drops whole small systems but never cuts a kept river
    big = river_network(
        hyd, land, layout, 5.0, slope, 3 * layout.hex_area_km2, 12 * layout.hex_area_km2
    )[0]
    assert big.sum() < river.sum() and not (big & ~river).any()


def test_desert_streams_lose_water_but_large_rivers_survive():
    layout, z = _plane()
    wet = _run(layout, z, runoff=200.0, precip=300.0, evap=2000.0)
    lossy = _run(layout, z, runoff=200.0, precip=300.0, evap=2000.0, loss=20.0)
    out = wet["routing"].outlet
    assert lossy["discharge_m3_s"][out].sum() < wet["discharge_m3_s"][out].sum()
    assert (lossy["channel_loss_m3_s"] >= 0).all()


# -- generated projects ------------------------------------------------------------------
def test_v4_project_layers_are_consistent(hist_project):
    p = hist_project
    h = p.hydro
    assert h is not None and h.validation["ok"], h.validation
    L = h.layers
    for name, arr in L.items():
        if np.issubdtype(arr.dtype, np.floating) and name not in ("lake_level_m", "lake_depth_m"):
            assert np.isfinite(arr).all(), name
    assert L["elevation_m"] is L["post_erosion_elevation_m"]
    assert L["pre_erosion_elevation_m"] is p.hist.layers["elevation_m"]  # v3 is untouched
    assert np.array_equal(L["land_mask"], L["elevation_m"] > h.sea_level_m)
    assert (L["sediment_thickness_m"] >= 0).all() and (L["v4_erosion_m"] >= 0).all()
    assert np.allclose(
        L["v4_delta_elevation_m"], L["elevation_m"] - L["pre_erosion_elevation_m"], atol=1e-2
    )
    assert p.layer("elevation_m") is L["elevation_m"]
    assert p.stage_layer("v3", "elevation_m") is p.hist.layers["elevation_m"]
    shares = h.summary["aridity_shares"]
    assert sum(shares.values()) == pytest.approx(1.0, abs=1e-3)
    b = h.budget
    assert abs(b["closure_error_m3"]) <= 1e-6 * max(b["fluvial_eroded_m3"], 1.0) + 1.0


def test_erosion_cuts_valleys_and_depends_on_rock(hist_project):
    p = hist_project
    L = p.hydro.layers
    land = L["land_mask"]
    # water cuts: erosion follows stream power (discharge and slope), not slope alone
    erosion = L["v4_erosion_m"][land]
    power = (np.sqrt(L["discharge_m3_s"]) * L["slope_m_per_km"])[land]
    by_power = spearmanr(erosion, power).correlation
    by_slope = spearmanr(erosion, L["slope_m_per_km"][land]).correlation
    assert by_power > 0.4 and by_power > by_slope, (by_power, by_slope)
    wc, _ = generate_climate(p)
    hist = p.hist
    graph = hex_graph(p.layout)
    no_uplift = replace(p.config, erosion=replace(p.config.erosion, continue_uplift=False))

    def run(resistance):
        return evolve(
            no_uplift,
            p.layout,
            graph,
            hist.layers["elevation_m"],
            hist.sea_level_m,
            np.full(hist.layers["rock_resistance"].shape, resistance),
            np.zeros(hist.layers["sediment_thickness_m"].shape),
            hist.layers["uplift_rate_mm_yr"],
            wc,
            None,
        )

    hard, soft = run(0.9), run(0.1)
    assert soft["erosion_total_m"].sum() > 2.0 * hard["erosion_total_m"].sum()
    z0 = hist.layers["elevation_m"].ravel().astype(np.float64)
    above = z0 > hist.sea_level_m
    assert soft["elevation"][above].mean() < z0[above].mean()  # without uplift land wears down
    for r in (hard, soft):
        assert (r["sediment_thickness_m"] >= 0).all()
        assert np.isfinite(r["elevation"]).all()


def test_v4_is_deterministic(hist_config, hist_project):
    again = generate(hist_config)
    assert again.hydro.hydro_hash() == hist_project.hydro.hydro_hash()
    assert again.hydro.rivers == hist_project.hydro.rivers


def test_river_threshold_reuses_climate_and_evolution(hist_config, hist_project):
    cfg = replace(
        hist_config,
        hydrology=replace(hist_config.hydrology, river_min_discharge_m3_s=60.0),
    )
    p = generate(cfg, reuse=hist_project)
    reused = set(p.stats["reused_stages"])
    assert {"v4_climate", "v4_evolution"} <= reused and "v4_network" not in reused
    assert np.array_equal(
        p.hydro.layers["post_erosion_elevation_m"], hist_project.hydro.layers["elevation_m"]
    )
    assert p.hydro.layers["river_mask"].sum() <= hist_project.hydro.layers["river_mask"].sum()
    assert p.hist.hist_hash() == hist_project.hist.hist_hash()


def test_climate_profile_recomputes_v4_but_not_the_geology(hist_config, hist_project):
    cfg = replace(hist_config, climate=replace(hist_config.climate, profile="dry"))
    dry = generate_v4(hist_project, cfg)
    assert not {"v4_climate", "v4_evolution"} & set(dry.stats["reused_stages"])
    assert dry.hist.hist_hash() == hist_project.hist.hist_hash()
    assert dry.geo.geo_hash() == hist_project.geo.geo_hash()
    land = dry.hydro.layers["land_mask"]
    assert (
        dry.hydro.layers["precipitation_mm_yr"][land].mean()
        < hist_project.hydro.layers["precipitation_mm_yr"][land].mean()
    )


def test_sea_level_change_recomputes_v4_from_the_new_coast(hist_project):
    p = derive_landmasses(hist_project, sea_level_m=150.0)
    assert p.hydro is not None and p.hydro.sea_level_m == 150.0 and p.hydro.validation["ok"]
    assert (
        p.hydro.summary["land_fraction_viewport"]
        < hist_project.hydro.summary["land_fraction_viewport"]
    )


def test_save_load_roundtrip_and_cache(tmp_path, hist_config, hist_project):
    path = save_project(hist_project, tmp_path / "p")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "hexworld-project-4" and "v4" in manifest
    geo = json.loads((path / "rivers.geojson").read_text(encoding="utf-8"))
    assert geo["type"] == "FeatureCollection"
    assert len(geo["features"]) == len(hist_project.hydro.rivers)
    back = load_project(path)
    assert back.hydro.hydro_hash() == hist_project.hydro.hydro_hash()
    assert back.hydro.rivers == hist_project.hydro.rivers
    assert back.hydro.validation["ok"]
    again = generate(hist_config, reuse=back)
    assert "v4_evolution" in again.stats["reused_stages"]
    assert again.hydro.hydro_hash() == hist_project.hydro.hydro_hash()


def test_stale_v4_data_is_never_loaded_silently(tmp_path, hist_project):
    path = save_project(hist_project, tmp_path / "p")
    doc = json.loads((path / "hydrology.json").read_text(encoding="utf-8"))
    doc["algorithm_versions"]["hydrology"] = "hydro-0"
    (path / "hydrology.json").write_text(json.dumps(doc), encoding="utf-8")
    back = load_project(path, verify=False)
    assert back.hydro is None and back.hist is not None
    assert any("NOT loaded" in w for w in back.warnings)


def test_v2_project_goes_to_v4_and_keeps_its_baseline(tmp_path):
    work = tmp_path / "proj"
    shutil.copytree(V2_FIXTURE, work)
    manifest_v2 = json.loads((V2_FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    v2 = load_project(work)
    ok, reason = views.VIEWS["hydrology"].availability(v2)
    assert not ok and reason
    v3 = generate_history(
        v2, replace(v2.config, history=HistoryParams(duration_ma=8.0, time_step_ma=2.0))
    )
    assert v3.hydro is None  # adding a history does not compute v4 silently
    assert not views.VIEWS["temperature"].availability(v3)[0]
    v4 = generate_v4(v3)
    assert v4.hydro is not None and v4.hydro.validation["ok"], v4.hydro.validation
    assert v4.content_hashes() == manifest_v2["content_hashes"]
    assert v4.geo.content_hashes() == manifest_v2["v2"]["content_hashes"]
    assert v4.hist.hist_hash() == v3.hist.hist_hash()
    save_project(v4, work)
    back = load_project(work)
    assert back.hydro.hydro_hash() == v4.hydro.hydro_hash()


def test_v4_views_draw_computed_data(hist_project):
    for view_id in ("temperature", "precipitation", "hydrology", "erosion"):
        spec = views.VIEWS[view_id]
        assert spec.status == "ready" and spec.availability(hist_project)[0]
        for mode in spec.modes:
            rgb = views.view_rgb(hist_project, view_id, mode.id)
            assert rgb.shape == (100, 100, 3)
            # a warm map may have no snow or ice at all: then two shades (land, sea) are honest
            minimum = 2 if mode.id in ("snow", "glacial") else 4
            assert len(np.unique(rgb.reshape(-1, 3), axis=0)) >= minimum, (view_id, mode.id)
            assert spec.legend(hist_project, mode.id)
    info = hist_project.hex_info(50, 50)
    assert "temperature_mean_c" in info and "v3_elevation_m" in info


def test_earth_comparison_reports_real_numbers(hist_project):
    from hexworld.realism import earth_comparison, report_markdown

    metrics = earth_comparison(hist_project)
    names = {m.name for m in metrics}
    assert {"средняя высота суши", "аридные земли (P/PET < 0,2)", "средний размыв суши"} <= names
    for m in metrics:
        assert m.value is None or np.isfinite(m.value), m.name
        assert m.source and m.earth
    checked = [m for m in metrics if m.verdict in ("в диапазоне", "вне диапазона")]
    assert checked and sum(m.verdict == "в диапазоне" for m in checked) >= len(checked) // 2
    assert "| Группа |" in report_markdown(hist_project, metrics)


def test_cli_climate_command_and_export(tmp_path, capsys):
    out = tmp_path / "p"
    args = ["generate", "--seed", "Cli9", "--width", "40", "--height", "30", "--hex-km", "150"]
    args += ["--mode", "world"]
    assert main([*args, "--duration-ma", "6", "--time-step-ma", "2", "--no-v4", "-q"]) == 0
    assert main([*args, "--duration-ma", "6", "--time-step-ma", "2", "-q", "-o", str(out)]) == 0
    text = capsys.readouterr().out
    assert "v4 validation   OK" in text and "no land in the window" not in text
    assert main(["climate", str(out), "--climate-profile", "wet", "--erosion-ma", "1"]) == 0
    text = capsys.readouterr().out
    assert "climate         wet" in text and "v1-v3 reused unchanged" in text
    assert load_project(out).config.climate.profile == "wet"
    exp = tmp_path / "exp"
    assert main(["export", str(out), "-o", str(exp)]) == 0
    assert (exp / "rivers.geojson").exists() and (exp / "river_systems.csv").exists()
    assert np.load(exp / "v4_discharge_m3_s.npy").shape == (30, 40)
    assert main(["render", str(out), "--view", "hydrology", "-o", str(tmp_path / "r.png")]) == 0
