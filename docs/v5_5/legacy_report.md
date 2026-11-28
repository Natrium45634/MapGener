# TZ 5.5 — legacy and migration (B02, B03)

* A config saved before 5.5 has no `biosphere` key: it loads with `BiosphereParams(enabled=False)`,
  so the world regenerates exactly as before (`Config.from_dict`, test
  `test_v55_integration::test_life_disabled_has_no_biota`).
* A project saved without the biosphere loads with `project.biosphere = None` («не
  рассчитано»); its files are byte-identical after loading (test
  `test_b02_b03_old_save_and_roundtrip`). Nothing is recomputed on open.
* `hexworld bio-generate OLD --out NEW` adds the biosphere to a **copy**; `--out` equal to
  the source is refused. The original is never overwritten.
* A stored biosphere is loaded only when its algorithm version and its stage key (the
  content of every stage it reads, the biosphere parameters, the registry checksums and
  the biology seed) match; otherwise a warning says why and the biosphere stays «not
  generated».
* World codes: a new world code includes `"biosphere":{"enabled":true}`; old codes decode
  with the biosphere off and reproduce old worlds.
* Genuine v3–v5.4 save fixtures are still missing (5.4.1 BLOCKED_MISSING_FIXTURE): the
  legacy read of 5.5 is tested on saves made by this build without the stage and on the
  v1/v2 genuine fixtures of `tests/golden` via the existing suite.
