# Source matrix — third-party solutions checked for HexWorld 4.2

Manual, section 15: `URL | commit | license | source_file | verified claim | adopt/decline | test`.

**How the sources were read.** Shallow, sparse clones (only the files listed) into a scratch
directory outside the repository. They were read as text and **never built, imported or
run**. No code was copied into HexWorld. Ideas taken over are re-derived in our own code,
with our own units and tests.

"Verified" means the claim was found in the source file named, not that the program was run.

## A. Read in the source code (commit pinned)

| Source | Commit | License | Files read | Claim verified in the code | Decision for HexWorld | Our test |
|---|---|---|---|---|---|---|
| [landlab/landlab](https://github.com/landlab/landlab) | `c8ca4332c333` | MIT | `src/landlab/components/stream_power/stream_power.py`, `src/landlab/components/flow_accum/flow_accumulator.py` | `StreamPowerEroder` uses `E = K A^m S^n − threshold` with an implicit solver along the receivers. The docs warn that a long step can be stable but inaccurate for transients. the `FlowAccumulator` docstring shows `HexModelGrid` examples | **Adopted (formula):** `E_c` threshold (`erosion.incision_threshold_m_ma`, default 0) in our implicit solver `implicit_incision`. **Adopted (test):** convergence when the step shrinks. **Declined:** using landlab as a dependency (one hex grid, own C ABI, Windows build) | `test_12_E01_*`, `test_E05_*` |
| [badlands-model/badlands](https://github.com/badlands-model/badlands) | `975f595ae313` | GPL-3.0 | `docs/proc.rst` (the process documentation; the Python sources were not cloned) | detachment-limited `E = κ (PA)^m S^n`; transport-limited capacity; incision laws as `f(Qs)` | **Concept only:** our sediment step is already "capacity → deposit the excess, pass the rest down", with a closed budget (E04). No GPL code was used | `test_v4` budget closure (E04) |
| [fastscape-lem/fastscape](https://github.com/fastscape-lem/fastscape) | `dbf9f3d7d6fd` | BSD-3-Clause | `fastscape/processes/channel.py`, `fastscape/processes/hillslope.py` | separate `k_coef_bedrock` / `k_coef_soil` chosen by the soil cover | **Already present:** soft sediment (`sediment > 20 m`) has its own `K`. A separate diffusivity was declined (one effective large-scale `κ`) | `test_erosion_cuts_valleys_and_depends_on_rock` |
| [r-barnes/richdem](https://github.com/r-barnes/richdem) | `9a1c97bb2529` | GPL-3.0 | `include/richdem/depressions/Barnes2014.hpp` | improved priority-flood; an `epsilon` variant (`nextafter`) gives flats a drainage direction | **Already present:** our priority-flood works on a copy (real heights untouched) and resolves flats by flood order + stable id. The epsilon fill was declined (it changes the routing surface by ulps; our order makes it unnecessary) | `test_inclined_plane_*`, `test_wet_depression_*` (H03–H05) |
| [pysheds/pysheds](https://github.com/pysheds/pysheds) | `cb7002f7443f` | GPL-3.0 | `pysheds/sgrid.py` (fill, flowdir, accumulation) | raster D8 semantics | **Declined:** D8 on a square raster is the wrong topology on hexes (manual 6.2). The semantics are kept as tests only | H02 in `test_v4` |
| [Mindwerks/worldengine](https://github.com/Mindwerks/worldengine) | `c359b6e70e6c` | MIT | `worldengine/simulations/precipitation.py`, `temperature.py`, `humidity.py` | precipitation = noise × a temperature curve (no transport, no mass budget); rain shadow by a separate pass | **Declined as a model:** the manual itself asks for a mass-conservative transport. Our transport with a closed budget replaces it | `test_C03_*` |
| [Jannchie/cartographer](https://github.com/Jannchie/cartographer) | `86aecd323a5e` | MIT | `src/gen/climate.ts`, `src/gen/hydrology.ts`, `src/gen/erosion.ts` | `climate.ts` is **annual only**: transport along x rows, noise 0.8–1.2. The README's "monsoon / seasonal precipitation" was **not found in the code** at this commit | **Declined:** the seasonal/monsoon scheme of HexWorld 4.2 is our own (section 5 of the manual), not taken from here. Recorded so the README claim is not repeated as verified | `test_C01_*` |
| [weigert/SimpleHydrology](https://github.com/weigert/SimpleHydrology) | `2709726934f0` | none in the repository | `source/water.h` | particle hydrology: droplets with momentum leave discharge and momentum maps | **Declined for now:** fine-scale detail with no mass budget per basin. A sub-hex river renderer might use it later (no license file → no reuse without the author's permission) | — |

## B. Checked only by name / remote, not in code

| Source | What was done | Status |
|---|---|---|
| [Mindwerks/plate-tectonics](https://github.com/Mindwerks/plate-tectonics) | `git ls-remote` only | **not checked.** The HexWorld plate model works and is tested (manual: "do not replace without comparison") |
| [weigert/SoilMachine](https://github.com/weigert/SoilMachine) | not cloned | not checked |
| [owenyuwono/demiurge](https://github.com/owenyuwono/demiurge) | not cloned | not checked |
| Red Blob Games, hex grids | documentation (used since v1.0) | the HexWorld geometry follows it; G01 tests |
| Barnes, Lehman & Mulla 2014 (Priority-Flood) | article, pseudocode | implemented since v4.0 on the 6-neighbour graph |
| Leopold & Maddock 1953 (hydraulic geometry) | literature | `W = 5 Q^0.5`, `D = 0.3 Q^0.4`: a start for tuning, documented as such |
| Calov & Greve 2005 (positive degree days, normal daily T) | literature | used for the expected positive degree days of a season |
| All projects of the manual's sections B and C (islands, coasts, GPU, Minecraft mods) | not opened | **not checked**; nothing taken from them. Coasts 4.1 rely on geographic classifications (Shepard, Valentin, Davies) |

Nothing here is claimed as "verified" unless the file in column 4 was actually read.
