# Legacy and migration report (G8)

| Check | Result |
|---|---|
| Legacy read v1 (`tests/data/v1_project`) | canonical payload hashes = manifest; files unchanged by reading (`test_t06_…[v1_project]`) |
| Legacy read v2 (`tests/data/v2_project`) | same (`test_t06_…[v2_project]`) |
| Upgrade copy v1 / v2 | load → save to a new folder → load → save again: identical content hashes; the original folder never written (`test_t07_…`) |
| Legacy regeneration | BLOCKED: the old solvers are not kept as runners; bitwise regeneration of old versions is not promised |
| v3 … v5.4 fixtures | BLOCKED_MISSING_FIXTURE (no genuine files) |
| Current golden | `tests/golden/current/gold1_v2.json`: exact hashes on Linux x86_64, key set on other platforms; save / reload keeps the hashes (`test_t09_…`) |
| Config migration | a config without `atmosphere` loads with the atmosphere off; without `earth` with the Earth system off; new fields (`boundary_profiles_json`, `latent_feedback`, `incoming_tc_per_year`) default to the previous behaviour |
| Algorithm versions | earth `hexworld-earth-2`, atmosphere `hexworld-atmosphere-5.4.1`: stored v5.3 / v5.4 Earth / atmosphere data of the previous versions is reported "NOT loaded (not generated)" instead of being shown as current (existing loader policy, see the inventory) |
