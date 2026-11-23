# Legacy fixture inventory (G0 / G8)

| Version | Genuine file in the repository | Origin | Read test | Status |
|---|---|---|---|---|
| v1 (hexworld-project-1) | `tests/data/v1_project/` | committed with HexWorld 2.0 (`9dc864c`, 14.11.2026), written by the v1 serializer | `tests/test_v2.py::test_v1_project_opens_and_is_not_changed`, `tests/test_v541_legacy.py` | available |
| v2 (hexworld-project-2, generator 2.0.0) | `tests/data/v2_project/` | committed with HexWorld 3.0 (`0be2475`, 15.11.2026) | `tests/test_portability.py`, `tests/test_v3.py`, `tests/test_v4.py`, `tests/test_v541_legacy.py` | available |
| v3.x … v5.4 (hexworld-project-3/4) | none | — | — | **BLOCKED_MISSING_FIXTURE**: no genuine file of these versions was kept; a file written now by the current serializer would not be a legacy fixture |

Policy found in `io.py` since v4: stored v4+ data produced by another algorithm version is
**not attached** (the project opens with v1–v3 intact and a warning); this predates 5.4.1
and conflicts with TZ 5.4.1 §11.1 "legacy read keeps stored arrays". Not changed in 5.4.1
(it would need read-only legacy layers through every stage); recorded as an open item.
Current-version reproducibility uses `tests/golden/current` (same seed / config / algorithm
→ same canonical hashes).
