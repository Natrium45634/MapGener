# Аудит входов v5.0 (ТЗ 5.0, gate G0)

Проверено на текущих исходниках, 22.11.2026. Значения статуса: `verified` (есть в коде и в
тестах), `approximate` (есть, но это оценка), `inferred` (выводится из других полей),
`missing` (нет).

| Поле ТЗ | Где на самом деле | Стадия | dtype / shape | Единицы | Статус | Кто использует в v5.0 |
|---|---|---|---|---|---|---|
| `elevation_m` | `hydro.layers["elevation_m"]` | v4 после v4.6 | float32 (H) | м | verified | ложе льда, морские глубины |
| `sea_level_m` | `hydro.sea_level_m` | v3/v4 | float | м | verified | глубина воды |
| `ocean_connected_mask` | `hydro.layers["ocean_mask"]` (связность v4.6) | v4.6/v4 | bool (H) | — | verified | морские участки; впадины ниже моря не море |
| `lake_id`, `lake_surface_level_m` | `hydro.layers` | v4 | int32 / float32 | —, м | verified | озёрные классы |
| `land_fraction` (подгекс) | `topo.layers["topo_dry_fraction"]` + участки | v4.7 | float32 | 0..1 | verified | доли суша / озеро / море |
| `rock_type` | `hist.layers["rock_type"]`, `env.parent_material_class` | v3 / v4.5 | uint8 | класс | verified | субстрат, молодая лава |
| `soil_depth_m`, `AWC_mm` | `env.layers`, участки `topo.units` | v4.5 / v4.7 | float32 | м, мм | verified | PFT (деревья), голые скалы |
| `T_monthly_c`, `P_monthly_mm`, `PET_monthly_mm`, `AET_monthly_mm` | `topo.units[...]` (12, U), иначе `env.monthly` | v4.7 / v4.5 | float32 | °C, мм/мес | verified | все признаки |
| `snow_swe_mm` | `topo.units["swe_mm"]` | v4.7 | float32 | мм в.э. | verified (сезонный) | — (v5.0 считает свой снег/фирн/лёд) |
| `waterlogging_months` | `env.layers` | v4.5 | float32 | мес. | approximate (по месяцам) | болота |
| `floodplain_potential` | `env.layers` | v4.5 | float32 | 0..1 | approximate | поймы |
| `permafrost_potential` | `env.layers`, `topo.units` | v4.5 / v4.7 | float32 | 0..1 | approximate | модификатор мерзлоты |
| `gdd_proxy` | `topo.units["gdd0/5/10"]` (суточная кривая) | v4.7 | float32 | °C·сут | verified | PFT |
| `slope/aspect` | `topo.units` | v4.7 | float32 | °, ° от севера | verified | скалы, тень склона |
| `topoclimate` | облака, туман, холодные карманы, граница леса | v4.7 | float32 | 0..1 | approximate | облачный лес, туманная пустыня |
| `TerrainUnit` | `topo.units` (CSR по гексам) | v4.7 | — | — | verified | основной масштаб T |
| `subhex area` | `unit_weight_hex` (плановая площадь) | v4.7 | float32 | доля гекса | verified | доли биомов |
| `soil_salinity_proxy` | `env.layers` | v4.5 | float32 | 0..1 | approximate (без баланса солей) | солончаки |
| `endorheic_mask`, `river_mask` | `hydro.layers` | v4 | bool | — | verified | солёные озёра, галерейные леса |
| SST, приливы, солёность моря | — | — | — | — | **missing** | мангры и кораллы не утверждаются (только потенциал) |
| месячный ветер, влажность | — (годовой ветер v4, синтетическая RH v4.7) | — | — | — | missing / approximate | не используются как сильный фактор |

## Выводы

- Вода по связности (ТЗ 4.6) уже работает: `ocean_mask` не равен `elevation < sea_level`.
  Каспиеподобные впадины — суша или озёра со своим уровнем.
- Граница леса 4.7 — термический потенциал. В v5.0 она входит как множитель пригодности
  деревьев, а не как готовый лес.
- AET из 4.5/4.7 относится к эталонному покрову. В v5.0 это **единственный** показатель
  водного стресса растений; P/PET — только диагностика (классы UNEP).
- Без v4.7 стадия работает в режиме `coarse_hex_fallback` (один участок на гекс), с флагом
  качества и предупреждением.
- Нет SST, приливов и морской солёности. Поэтому мангры, кораллы и морской лёд — только
  потенциал, о чём сказано в `provenance`.

Запуск тестов перед работой: `pytest` (Linux, Python 3.13, numpy 2.5, scipy 1.18) — зелёный;
`pytest -m slow` — зелёный; CI Windows 3.12/3.13 — зелёный на коммите 4.7.0.
