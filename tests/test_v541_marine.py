"""TZ 5.4.1 G6: marine habitat axes and the biome registry (v5_4_1::T67-T75, N19, N20)."""

from __future__ import annotations

import numpy as np
import pytest

from hexworld.biomes import marine as mar


def test_n19_t72_light_with_depth_and_ice():
    assert mar.euphotic_depth(0.1) == pytest.approx(46.0517, rel=1e-5)
    assert mar.par_fraction(10.0, 0.1) * 100 == pytest.approx(36.78794, rel=1e-6)
    assert mar.par_fraction(100.0, 0.1) * 100 == pytest.approx(0.00453999, rel=1e-5)
    assert mar.par_fraction(0.0, 0.1, ice_c=1.0) == 0.0  # opaque ice
    assert mar.par_fraction(0.0, 0.1, ice_c=1.0, transmission=0.1) == pytest.approx(0.1)


def test_n20_degree_heating_weeks():
    sst = np.full(84, 28.0)
    sst[-7:] = 31.0  # HotSpot 2 above a maximum monthly mean of 29
    assert mar.degree_heating_weeks(sst, 29.0) == pytest.approx(2.0)
    later = np.concatenate([sst, [27.0]])  # one cool day does not reset the window
    assert mar.degree_heating_weeks(later, 29.0) == pytest.approx(2.0)
    rolled = np.concatenate([sst, np.full(84, 27.0)])
    assert mar.degree_heating_weeks(rolled, 29.0) == 0.0  # old days leave the window


def _line(n=8, depth=None, sst=26.0, sst_range=2.0, ice=0.0, up=0.0, mouth=None):
    depth = np.full(n, 20.0) if depth is None else np.asarray(depth, float)
    th = 2 * np.pi * np.arange(12) / 12
    sst_m = sst + sst_range * np.cos(th)[:, None] * np.ones((12, n))
    ice_m = np.full((12, n), ice)
    nbr = np.full((n, 6), -1, dtype=np.int64)
    nbr[:-1, 0] = np.arange(1, n)
    nbr[1:, 3] = np.arange(n - 1)
    mouth = np.zeros(n, bool) if mouth is None else mouth
    return mar.habitats(
        depth,
        np.ones(n, bool),
        sst_m,
        ice_m,
        np.full(n, up),
        mouth,
        np.zeros(n, bool),
        np.zeros(n, bool),
        np.zeros(n, bool),
        nbr,
    )


def test_t74_coral_needs_light_and_clear_water_not_only_warmth():
    shallow = _line()
    assert (shallow["marine_coral_reef_potential"] > 0).all()
    deep = _line(depth=np.full(8, 3000.0))
    assert (deep["marine_coral_reef_potential"] == 0).all()  # warm SST alone is not enough
    mouth = np.zeros(8, bool)
    mouth[3] = True
    turbid = _line(mouth=mouth)
    assert turbid["marine_coral_reef_potential"][2:5].max() == 0  # river turbidity
    cold = _line(sst=14.0)
    assert (cold["marine_coral_reef_potential"] == 0).all()


def test_t71_pelagic_and_benthic_axes_are_separate():
    h = _line(depth=[50, 150, 600, 1500, 3000, 5000, 20, 20], sst=8.0)
    assert list(h["marine_benthic_zone"][:6]) == [1, 1, 2, 2, 3, 3]
    assert (h["marine_pelagic_zone"][:6] == 3).all()  # one cold water column over all depths
    ice = _line(sst=-1.0, ice=0.5)
    assert (ice["marine_pelagic_zone"] == 4).all()


def test_productivity_follows_nutrients_not_latitude():
    poor = _line(up=0.0, sst_range=0.5)
    rich = _line(up=0.8, sst_range=0.5)
    assert (rich["marine_productivity_potential"] > poor["marine_productivity_potential"]).all()
    assert (rich["marine_nutrient_supply_proxy"] <= 1.0).all()


def test_t75_isolated_basins_are_separate_regions_and_ids_are_stable():
    shape = (6, 12)
    ocean = np.zeros(shape, bool)
    ocean[:, :5] = True
    ocean[:, 7:] = True  # two basins, same SST, separated by land
    sst = np.full(shape, 20.0)
    ids = mar.ecoregions_fast(ocean, sst, np.zeros(shape), np.full(shape, 1000.0), shape)
    assert ids[0, 0] != ids[0, 11]
    assert (ids[~ocean] == -1).all()
    again = mar.ecoregions_fast(ocean, sst, np.zeros(shape), np.full(shape, 1000.0), shape)
    assert np.array_equal(ids, again)
    assert ids[0, 0] == 0  # numbered from the first hex


def test_t67_registry_keeps_every_old_id():
    from hexworld.biomes.registry import BIOMES, INDEX, validate_registry

    assert not validate_registry()
    ids = [b.id for b in BIOMES]
    assert len(ids) == len(set(ids)) == 110
    assert INDEX["marine_hadal_trench"] == 100  # v5.0 codes unchanged, extensions appended


def test_stage_marine_layers(hist_project):
    B = hist_project.biomes.layers
    if "marine_pelagic_zone" not in B:
        pytest.skip("no v5.3 ocean in this project")
    ocean = hist_project.hydro.layers["ocean_mask"].astype(bool)
    assert (B["marine_pelagic_zone"][~ocean] == 0).all()
    assert (B["marine_ecoregion_id"][~ocean] == -1).all()
    assert (B["marine_ecoregion_id"][ocean] >= 0).all()
    pp = B["marine_productivity_potential"][ocean]
    assert np.nanmin(pp) >= 0.0 and np.nanmax(pp) <= 1.0
