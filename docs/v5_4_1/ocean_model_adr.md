# ADR: ocean solver and SST source (5.4.1)

**Kept:** the v5.3 ocean on the world raster (connected-sea mask, Stommel transport
streamfunction on the cell corners — exactly divergence-free face transports, western
boundary layer; implicit mixed-layer heat with upwind advection and eddy diffusion; zero-
layer sea ice with drift). Its 5.3 tests (gyre with a western boundary current, closed
basin, straits, disconnected basins, heat conservation, upwelling, ice) pass unchanged.

**Replaced:**
1. the restoring of the SST to an Earth zonal SST table and the hidden zonal-mean
   calibration → air-sea exchange `λ (T_A − SST)` with the planet's seasonal energy-balance
   model `T_A(lat, month)` (`earth/ebm.py`), λ = 35 W m⁻² K⁻¹;
2. the fixed subsurface temperature `T_eq − 6 °C` (an infinite cold sink) → a per-basin
   subsurface reservoir with closed heat;
3. the f-clamped Ekman transport → the drag-regularised slab response (finite at f = 0,
   sign kept in the south).

**Added:** open ocean sides (boundary profiles): two-way exchange with an external sea at a
prescribed temperature, without a net volume flux (booked as `boundary_heat_j`).

**Single SST:** `run_ocean` is the only producer; the 5.4 coupled iteration re-runs it
with the new monthly winds and replaces the result.

**Mesh / projection:** the world raster (flat km, latitude per row from the map's latitude
setting); the hex window samples it. A hex-edge ocean solver was not introduced: the
existing raster solver passes the conservation tests and the hex sampling is a display /
downstream step.

**Not done (O2/O3):** two-layer dynamics with salinity, full-depth / thermohaline
circulation (5.7).
