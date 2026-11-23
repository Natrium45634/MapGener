# Units, constants and formulas

Everything is SI and float64 unless a layer says otherwise. Masses are in kg, volumes in m³
and rates in kg/s or m³/s. Event time is in years before present.

DRE volume = mass / magma density. Bulk tephra volume = tephra mass / 1000 kg/m³. The two
are never mixed.

## Constants

| Name | Value | Note |
|---|---|---|
| L_f (fusion of ice) | 334 000 J/kg | |
| ρ ice / water / sea water | 917 / 1000 / 1025 kg/m³ | |
| c_p magma | 1200 J/(kg·K) | |
| L crystallisation | 4·10⁵ J/kg | |
| ρ tephra (bulk) | 1000 kg/m³ | |
| ρ PDC deposit | 1400 kg/m³ | |
| SO₂ per S | 64/32 | |

## Formulas ([H] = heuristic, coefficients are configuration)

**Viscosity [H]**

    log10 η = 2.0 + 0.25(SiO₂ − 50) − 0.006(T − 1150) + 1.5X − 0.25 H₂O

This gives basalt 10¹–10³, andesite 10⁵–10⁶ and hydrous rhyolite 10⁸–10¹⁰ Pa·s.

**Reservoir**

    dM/dt = (1 − i)·Ṁ_in − M/τ,   τ = 5000 yr

- The intrusive share `i` is 0.9 at ridges, 0.8 at arcs, 0.55 at oceanic hotspots, 0.9 at
  continental hotspots and 0.85 at rifts.
- The equation is solved exactly between events.
- The budget `initial + supplied − erupted − intruded − solidified − melt` closes to
  ~1e-15.

**Eruption rate [H]**

    λ = λ₀(edifice) · clip((Ṁ/500)^0.3, 0.2, 3) · f_status · scale · clip(P, 0.05, 3)²

- `P = M / M_crit`.
- Overpressure relief applies when `P > 1.3`: the eruption also taps half of the excess
  melt.

**Volume** — a Pareto draw (`b` = 0.7–1.0) capped by the edifice maximum and by half of
the available melt.

**Explosive tendency [H]**

    E = σ(c0 + c1 log η + c2 H₂O_retained + c3 ascent + c4 water − c5 open_conduit)

The coefficients are c0 = −4.4, c1 = 0.62, c2 = 0.55, c3 = 0.35, c4 = 1.4 and c5 = 1.6.

**Mass partition**

    M = M_lava + M_tephra + M_gas

`M_gas = M · volatiles · degassing`. The tephra share depends on the style.

**Column height** — Mastin et al. 2009:

    H = 2.00 · Q_DRE^0.241 km

The column is 0 for a deep-submarine vent.

**VEI** — from the bulk tephra volume: 1e4, 1e6, 1e7 … 1e12 m³.

**Lava [H]**

    h_res(d) = h₀ · exp(d / L_cool),   L_cool = 1.6 √Q km

- `h₀` is 1.5 m for basalt and 12 m for andesite.
- `L_cool` is scaled by 0.25–0.35 under water or ice.
- Below sea level the residual thickness is ×1/0.4.

**Tephra** — upwind face fluxes `m·max(0, u·n)·dt·L/A`, diffusion `K = 3000 m²/s` and
settling `1 − e^(−dt/t_fall)`. The CFL condition keeps the summed outflow share below 0.8.

**PDC energy line** — Malin & Sheridan:

    z_vent + H_c − μ d > z,   μ = clip(0.35 V^−0.06, 0.08, 0.35)

**Lahar** — Iverson et al. 1998:

    V = min(W/(1 − c), S/c),   c = 0.55,   A_inundated = 200 V^(2/3)

**Melt (energy limited)**

    M_melt = ε · M · (c_p T + L_c) / L_f

ε is 0.15 for an eruption on snow or ice and 0.6 under ice (contact). The melt is capped
by the ice that is there.

**Outburst peak** — Clague & Mathews: `Q = 75 (V/10⁶)^0.67` m³/s.

**Climate pulse [H]**

    ΔT = −0.08 K · (SO₂_strat / Tg)^0.75,   e-fold 1 yr

Pinatubo gives about −0.6 K. The pulse is a record only.

**Recovery [H]**

    t_rec = 150 (1 + 4 cold)(1 + 2.5 dry) yr

`cold = clip((22 − T)/30, 0, 1)` and `dry = clip((1200 − P)/1000, 0, 1)`.
