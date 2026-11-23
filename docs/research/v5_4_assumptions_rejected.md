# TZ 5.4 — assumptions and rejected approaches

| Rejected | Why | Used instead |
|---|---|---|
| A fixed rain parallel for the ITCZ | TZ 6.1; segments and gaps must be possible | `-div Q` × warmth × moisture inside the Hadley band, connected components (T38) |
| `monsoon = wind reversal` | reversal without moisture is not a monsoon (T40, N18) | suitability needs a rain range, a summer share and a land–sea contrast or onshore flux |
| `SST > 26.5 → storm` | TZ 8.1, R12 | six-factor genesis index, Poisson rate, storms may be absent |
| Geostrophy with `1/f` | singular at the equator (T13, N16) | friction-pressure-Coriolis balance with r > 0 |
| A second SST over the sea | one SST (T10) | v5.3 ocean; coupled re-run instead |
| Storm rain added to the norm | double count (TZ 6.2) | storm share moved out of the convective component; event rain from a shared pool |
| Independent random weather per hex | T63 | correlated uniforms (600 km, AR(1) in time) |
| Earth ENSO teleconnection maps | arbitrary continents (TZ 10.2, R23) | ocean mode only with a planetary domain and an equatorial upwelling tongue; response computed by re-solving the vapour budget |
| Storm surge flooding without hydrodynamics | TZ 8.5 | `tc_surge_potential_index` only |
| Lat/lon in exports | the flat map has no CRS | world km (`geometry_world_km`) |
| Every pressure level on every hex | memory (TZ 21) | coarse atmosphere grid + on-demand sampling |
| Normalising every region to one world rain total | TZ 6.2 | the 5.4 normals use physical units with fixed calibration constants; no per-world scale |

Open assumptions (documented, not hidden): monthly quasi-steady vapour budget; mid-level
humidity = column RH (proxy); fixed Brunt–Väisälä frequency in the Eady rate; ETC and AR
climatologies are proxies; TC intensity is an upper-bound ODE, not Emanuel PI; the
circulation does not respond to latent heating except via the coupled ocean iteration.
