# TZ 5.4 — water and energy budget

## Vapour column (climate normals)

Finite volumes on the atmosphere grid; one implicit step per month (the vapour residence
time, about a week, is shorter than a month, so the month is a quasi-steady normal — TZ
3.2 allows stationary iteration with a documented residual):

```text
(W_m - W_(m-1)) / dt_m + div(u_q W_m) = E_m - P_m
E_sea  = rho C_E |U| (q_s(SST) - (a + b W / W_sat) q_s(T_a)) (1 - ice)   (linear in W)
E_land = min(PET, Budyko-Fu share of the rain supply)                    (recycling)
P      = (k_bg + k_conv + k_oro + k_front) W                              (each >= 0)
```

`u_q = 0.6 u_sfc + 0.4 u_850` (vapour-weighted). Rates depend on `W` through the column
relative humidity, so each month runs 2–4 Picard iterations (one sparse LU each) with
the rates of the previous iterate; the booked terms are computed with exactly the rates
of the last linear solve, so the ledger is exact up to the LU precision.

Per month the ledger (kg over the domain, `AtmosphereData.summary["budget_months"]`):

```text
storage_change = evaporation - precipitation + import - export + residual
```

Measured: worst monthly relative residual ~1e-16 … 1e-17 (validator threshold 1e-8;
tests use 1e-9, `tests/fixtures/v5_4/numerics.json`). Negative evaporation (dew / fog
onto water colder than the air) is allowed and booked as such; negative `W` and negative
rain components are invalid (validator).

Boundary: `isolated_approximation` / `planetary_background` use zero-gradient inflow (the
boundary column flows in), booked as import (`boundary_forcing = synthetic_zero_gradient`).
A specified boundary (`x_bc`, TZ `specified_boundary`) is an API argument and is booked the
same way (T28).

## Rain components

`P = P_conv + P_oro + P_front + P_storm + P_background`, each ≥ 0 and limited by the
vapour actually present. `P_storm` is not added: the TC climatology moves the expected TC
rain (passages × rain per passage) out of the convective component (at most 60 % of it),
so the total is unchanged (no double count; `tc_rain_attribution` in the summary).

## Downstream

- `precipitation_coupling = seasonality_only` (default): each hex keeps its calibrated v4
  annual total, the 12-month course is the normals' (`atmosphere_5_4_monthly_normals`).
  The atmosphere's own rain and the hydrology's annual rain may then differ locally;
  this is a declared approximation (`world_climate.precipitation_coupling`).
- `full`: the season sums of the normals become the world climate's seasonal rain
  (conservative upsampling: block totals of the atmosphere grid kept exactly), so the
  hydrology, soils and biomes see the 5.4 rain. Measured effect in
  `v5_4_benchmarks_and_coverage.md`.

## Weather year

```text
P_actual(month) = generator(background) + events
background      = (P_norm - P_storm - 0.5 P_front) x interannual factor
events          = TC / ETC fronts / AR rain, each limited by the vapour it can draw and
                  by one shared monthly pool (W x days / 8 d) — overlapping events never
                  spend the same water (T60)
```

`P_actual ≠ P_norm` is allowed (a sample, `unconditioned`); the ledger
`generator + events = actual` holds to float32 precision (N14). `conditioned` nudges each
month toward the background with a reported factor (clipped to 0.5–2).

## Energy

The surface temperature uses a one-box response to the planet's insolation cycle
(first two harmonics, lag `atan(k omega tau)`, tau 30–50 d on land); the ocean is the v5.3
mixed layer (its heat budget closes to ~1e-16, v5.3 validator). 5.4 does not solve a
second surface energy budget over the sea (one SST, T10). Not modelled: cloud radiative
feedbacks and full moist dynamics. Latent heating enters once: the convective rain above
the row mean lowers the SLP by 0.5 hPa per mm/day (smoothed 800 km), the winds and the
vapour budget are then solved again (`atmosphere.latent_feedback`, off at `fast`).
