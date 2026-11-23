# Boundary profiles (`hexworld-boundary-1`)

Stored in `atmosphere.boundary_profiles_json` of the config (canonical JSON: sorted keys,
6 significant digits — the same text from the GUI editor, a CLI config file and a saved
project; `boundary.profile_hash`). Modes of `atmosphere.domain_mode`:
`isolated_approximation` (no profiles: zero-gradient synthetic inflow, labelled),
`planetary_background` (world mode), `custom_region` (a profile for every side is
required), `closed_test_basin` (no exchange at all). `nested_region` (parent world) is
**not available** in this build.

| Field | Unit | Meaning |
|---|---|---|
| `boundary_id`, `sides` | — | the group of boundary faces (`west`, `east`, `north`, `south`) |
| `atm_mode` | — | `interior` (zero gradient), `closed`, `prescribed_state`, `prescribed_flux` |
| `t_air_c` | °C ×12 | external air temperature (state) |
| `q_kgkg` or `rh` | kg/kg or 0..1 ×12 | external humidity; RH is converted with T and `p_pa` |
| `p_pa` | Pa | pressure of the state |
| `moisture_flux_kg_m_s` | kg m⁻¹ s⁻¹ ×12, + inward | direct flux (excludes a state) |
| `ocn_mode` | — | `closed` or `exchange` |
| `exchange_m3s`, `ocean_t_c` | m³/s ≥ 0, °C ×12 | two-way exchange with the external sea, no net volume |
| `source`, `quality`, `preset` | — | provenance |

Physics: the column model of the atmosphere imports `L u_n W_ext` through an inflow face
(`W_ext = 1.2 · 2200 · q`, the column of the model's scale height), exports `L u_n W` through
an outflow face (upwind); closed faces carry nothing; a direct flux is a booked source on
the boundary cells. The ocean exchange adds `ρ c g (T_ext − T)` on the sea cells of that
side, booked as `boundary_heat_j`; a land edge carries no ocean exchange. Corner cells
take one side's donor (documented simplification). Presets: warm / cold ocean, wet
westerly, dry continental, mountain barrier (closed), interior.
