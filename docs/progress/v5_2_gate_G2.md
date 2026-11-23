# TZ 5.2 — gate G2

- **implemented:** static soil fertility: C/N/P/K stocks, CEC, pH, salts, suitability
- **tested:** test_v52_soil.py N01-N06, N10, S01-S05, S13-S16, S19-S22, S24, scenes
- **failed:** none at the commit of this file (see `docs/v5_2/acceptance_report.md` for totals)
- **blocked:** —
- **benchmarks:** `docs/v5_2/performance_report.md`
- **limitations:** see the open items of `docs/v5_2/acceptance_report.md`
- **source revisions:** RothC / Miami / Adams / Maas-Hoffman formulas (not inspected online)
- **ADR:** RothC pools describe the topsoil; the profile follows a depth distribution
