# TZ 5.2 — gate G3

- **implemented:** monthly / annual budgets with bounded fluxes, spin-up
- **tested:** S06-S12, S17, S18, N07-N09, test_s23_stage_ledgers
- **failed:** none at the commit of this file (see `docs/v5_2/acceptance_report.md` for totals)
- **blocked:** —
- **benchmarks:** `docs/v5_2/performance_report.md`
- **limitations:** see the open items of `docs/v5_2/acceptance_report.md`
- **source revisions:** —
- **ADR:** static state = analytic equilibrium + soil-age approach; implicit Euler on refining geometric steps (first order, S08)
