# TZ 5.4 — architecture, data flow and units

## Modules (`src/hexworld/atmosphere/`)

| Module | Content |
|---|---|
| `fv.py` | edge mesh (`raster_mesh` 4 faces, `hex_mesh` 6 faces), upwind operator `dX/dt = -D X + b`, CFL substeps, ledger, `validate_budget` (N10, N11, T22, T23, T29) |
| `circulation.py` | monthly temperature with land/sea inertia, thermal equator, Hadley edge (Held–Hou scaling), SLP, frictional winds (surface, 850), thermal wind (500, 250), `wind_at` with quality codes |
| `moisture.py` | column vapour `W`, implicit monthly finite-volume budget, bulk evaporation, land recycling, rain components |
| `normals.py` | coarse atmosphere grid, conservative transfer back to the raster, ITCZ score and segments, `AtmosphereNormals` (immutable snapshot, `source_hash`) |
| `monsoon.py` | seasonality, wettest 3 months, range, summer share, reversal score, onset / withdrawal, peaks, suitability, regimes, daily onset with false-start rejection |
| `storms.py` | TC genesis factors, Poisson rate, steered tracks, Holland profile, Eady growth, frontogenesis, AR detection, climatology ensemble |
| `weather.py` | Markov + Gamma weather generator with spatially / temporally correlated uniforms, TC / ETC (with fronts) / AR events, shared vapour pool, droughts, interannual mode |
| `stage.py` | coupling inside the world climate (normals, coupled ocean iteration, raster outputs) |
| `../pipeline_v54.py` | hex stage `v54_atmosphere`: diagnostics, storm climatology, weather year, `windAt` on world points, `applyWeatherEffects` |

## Order (TZ 11.2)

```text
v4 world climate (world raster)
  v5.3 coupler: ocean, sea ice, land temperature        (unchanged)
  v4 seasonal moisture loop                             (unchanged; spin-up / v4 totals)
  v5.4 normals on the atmosphere grid (factor 1..n)
    12 months: T (inertia), SLP, winds sfc/850/500/250, W, Q, P components, ITCZ
  coupled iteration (atmosphere.coupled_iterations, default 1):
    ocean re-run with the normals' monthly surface winds -> normals with the new SST
    report: rms dSST, mean rel. dP, mean rel. dwind; CONVERGED / NOT_CONVERGED
  publish monthly rain and T on the raster
v4 hydrology (annual, unchanged API)
v4.5 months: precipitation_coupling = seasonality_only (default): hex annual v4 total x
             monthly shares of the normals; full: season totals of the normals
v5.3 hex fields
v5.4 hex stage: monsoon diagnostics, storms, optional weather year (delta layers)
v4.7, v5.1 (tephra reads windAt), v5.0
```

The month or level selected in the GUI only samples the stored snapshot (T79).

## Grids and memory (TZ 21)

The atmosphere grid is the world raster coarsened by an integer factor so that it holds
~12k / 30k / 80k cells (`fast` / `balanced` / `high`). Stored per month on that grid:
SLP, surface pressure, T (sea level and surface), u/v on 4 levels, level heights, W, IVT,
convergence, E, P, 5 rain components, column RH, ITCZ score, storm fields — float32.
For a 512×458 raster at `balanced` (factor 3, ~26k cells) this is ~45–50 MB uncompressed (~36 monthly float32 fields).
Hex layers are only 2-D diagnostics; monthly views sample the grid on demand.

## Units and conventions

| Quantity | Unit | Note |
|---|---|---|
| `W` column vapour | kg/m² = mm of water | |
| `Q`, IVT | kg/(m s) | east / north components |
| `E`, `P`, components | mm per month in snapshots; kg/(m² s) inside the solver | |
| moisture convergence | kg/(m² s) | `-div Q` |
| winds | m/s, (east, **north**) in the atmosphere; hex / tephra / v5.3 layers keep map axes (+ south) | converted at the boundary |
| pressure | Pa | levels 85000 / 50000 / 25000 modelled; 92500 / 70000 log-p interpolated; 10000 / 5000 extrapolated |
| positions of events | world km (x east, y south) | never lat/lon (the flat map has no CRS) |
| time | model calendar: 31, 28, … days (Earth); equal months otherwise | T70 |

## API mapping (TZ 12)

| TZ call | Implementation |
|---|---|
| `inspectAtmosphereInputs` | `docs/v5_4_baseline_audit.md`; `WorldClimate.balance["atmosphere"]["status"] = blocked` without v5.3 (T01) |
| `buildSeasonalCirculation` | `atmosphere.normals.build` |
| `calculateMoistureTransport` | `atmosphere.moisture.solve` (ledger per month) |
| `findITCZAndMonsoonZones` | `normals.itcz`, `pipeline_v54.grid_diagnostics`, `monsoon.*` |
| `calculateStormClimatology` | `storms.climatology` |
| `windAt` | `pipeline_v54.wind_at_world(atm, x_km, y_km, month, pressure_pa= / altitude_m=)` → (u, v, quality) |
| `generateWeatherYear` | `weather.generate` |
| `stepWeatherEvent` | `storms.run_track` (TC), `weather.etc_events` (ETC + fronts) |
| `applyWeatherEffects` | `pipeline_v54.apply_weather_effects` → delta dict, committed with the stage object |
| `exportAtmosphericFields` | project save: `atmosphere_fields.npz`, `atmosphere_grid.npz`, `atmosphere.json` (events with metric geometry) |
| `validateAtmosphericBudget` | `stages/validate.py::validate_atmosphere`, `fv.validate_budget` |
