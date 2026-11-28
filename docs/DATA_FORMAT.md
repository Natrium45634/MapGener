# Формат проекта HexWorld (`hexworld-project-4`)

Читаются четыре схемы: `hexworld-project-1` (только v1.0), `hexworld-project-2`, `hexworld-project-3` и `hexworld-project-4`.

Проект — папка с файлами:

| Файл | Содержимое |
|---|---|
| `config.json` | сид, размеры, физический масштаб, режим, параметры всех стадий |
| `plates.json` | `world_id`, единицы, описание моделей плит и коры, список плит |
| `fields.npz` | слои гексов, массивы `[row, col]` |
| `boundaries.npz` | граф граничных рёбер, структура массивов |
| `boundaries.json` | описание массивов `boundaries.npz` и сводка по парам плит |
| `continental_fields.npz` | **v2.0:** слои материков и рельефа, мировые растры |
| `continents.json` | **v2.0:** уровень моря, сводка, блоки, материки, ключи стадий |
| `manifest.json` | версии, система координат, единицы, ключи стадий, контрольные хэши |
| `history_fields.npz` | **v3.0:** слои выбранной эпохи и мировые растры этой эпохи |
| `history.json` | **v3.0:** эпохи, сводка, вулканы, горячие точки, материки эпохи, баланс эрозии |
| `events.json` | **v3.0:** журнал геологических событий |
| `checkpoints.npz` | **v3.0:** состояние породы на каждом сохранённом снимке |
| `mountain_plan.json` | **v3.0:** горные системы, хребты, вершины, перевалы, долины |
| `hydro_fields.npz` | **v4.0:** слои климата, гидрологии и эрозии |
| `hydrology.json` | **v4.0:** версии, единицы, происхождение, сводка, баланс наносов, реки, озёра, бассейны, материки итоговой поверхности |
| `rivers.geojson` | **v4.0:** сегменты рек как LineString в мировых км (дополнительное представление; источник — `hydrology.json`) |
| `config.v1.json`, `manifest.v1.json`, `*.v2.json`, `*.v3.json` | копии исходных файлов, если проект старой схемы был дополнен |

PNG — только изображение. Оно никогда не читается как данные.

## Координаты

- Мир: плоскость, км. `+x` — восток, `+y` — юг. Начало координат произвольно.
- Гексы: flat-top. Расстояние между центрами соседей — `hex_spacing_km`, радиус описанной
  окружности `R = hex_spacing_km / √3`.
- Хранение: odd-q offset, массив `[row, col]`; нечётные столбцы сдвинуты на полгекса к `+y`.
- Центр гекса: `x = x0 + R + 1.5·R·col`, `y = y0 + s/2 + s·(row + 0.5·(col & 1))`,
  где `(x0, y0)` — `viewport_origin_km`, `s` — `hex_spacing_km`.
- Axial: `q = col`, `r = row − (col − (col & 1)) / 2`. Cube: `(q, −q − r, r)`.
- Направления соседей 0..5 в axial: `(1,0) (1,−1) (0,−1) (−1,0) (−1,1) (0,1)`.

В `manifest.json → coordinates.geometry` записаны `simulation_bounds_km`,
`viewport_bounds_km`, `model_bounds_km` (область симуляции с ореолом), размеры сетки и гекса.

## `fields.npz`

| Слой | Тип | Единицы | Смысл |
|---|---|---|---|
| `plate_id` | int32 | — | ID плиты, индекс в `plates.json → plates` |
| `crust_type` | uint8 | — | 0 океаническая, 1 континентальная; не маска суши |
| `continentality_km` | float32 | км | знаковое псевдорасстояние до границы типов коры, `> 0` — континентальная |
| `velocity_x`, `velocity_y` | float32 | мм/год | скорость плиты в центре гекса |
| `boundary_type` | uint8 | — | 0 нет, 1 сближение, 2 расхождение, 3 сдвиг, 4 спокойная |
| `boundary_distance_km` | float32 | км | расстояние до ближайшей границы, не больше `fields.max_influence_km` |
| `uplift`, `rifting`, `subduction`, `volcanism`, `seismicity` | float32 | 0..1 | диагностические потенциалы |

## `plates.json`

Для каждой плиты: `id`, `name`, `color`, `size_class`, `halo` (центр вне области симуляции),
`site_km`, `weight_km`, `area_km2` (в области симуляции), `centroid_km`,
`continental_fraction`, `oceanic_fraction`, `velocity_mm_yr`, `speed_mm_yr`, `omega_deg_myr`,
`rotation_pole_km`, `neighbors`, `visible_hexes`, `visible_area_km2`,
`visible_continental_fraction`.

Скорость в точке `p`: `v = (vx, vy) + ω × (p − pole)`, то есть
`(vx − ω·ry, vy + ω·rx)` при `ω` в рад/млн лет. 1 км/млн лет = 1 мм/год.
Площадь, центр, доли коры и соседи измерены на опорном растре мира и не зависят от окна.

## `boundaries.npz`

Одна запись на каждое общее ребро двух гексов разных плит; оба гекса лежат в окне.
Всегда `plate_a < plate_b`.

| Массив | Тип | Смысл |
|---|---|---|
| `col_a`, `row_a`, `col_b`, `row_b` | int32 | гексы со стороны A и B |
| `direction` | uint8 | направление от A к B, 0..5 |
| `plate_a`, `plate_b` | int32 | ID плит |
| `mid_x`, `mid_y` | float64, км | середина ребра |
| `normal_x`, `normal_y` | float32 | единичная нормаль геологической границы, из A в B |
| `vrel_x`, `vrel_y` | float32, мм/год | `v_B − v_A` в середине ребра |
| `vn` | float32, мм/год | `v_rel · n`; `< 0` сближение, `> 0` расхождение |
| `vt` | float32, мм/год | `v_rel · t`, `t = (−n_y, n_x)`; `> 0` правый сдвиг |
| `boundary_type` | uint8 | как в слое `boundary_type` |
| `intensity` | float32 | преобладающая составляющая, делённая на `boundaries.reference_speed_mm_yr` |
| `crust_a`, `crust_b` | uint8 | тип коры по сторонам |
| `subducting` | uint8 | подсказка полярности: 0 нет, 1 плита A, 2 плита B |

Длина каждого ребра равна `R` и записана в `boundaries.json → edge_length_km`.
Нормаль берётся из непрерывной модели плит, а не из зигзага рёбер гексов.

## `continental_fields.npz` (v2.0)

Массивы `[row, col]` той же сетки, что и `fields.npz`.

| Слой | Тип | Единицы | Смысл |
|---|---|---|---|
| `continental_fraction` | float32 | 0..1 | непрерывная доля континентальной коры |
| `continental_margin_km` | float32 | км | знаковое поле «расстояние до края коры», `> 0` — континентальная |
| `continental_block_id` | int32 | — | связный блок континентальной коры (`continental_fraction ≥ 0.5`), −1 — нет блока |
| `macro_elevation_m` | float32 | м | абсолютная макровысота в единой вертикальной системе |
| `tectonic_relief_m` | float32 | м | тектоническая часть макровысоты: области поднятия и впадин |
| `land_mask` | bool | — | `macro_elevation_m > sea_level_m` |
| `landmass_id` | int32 | — | связная надводная суша, −1 — под водой |
| `shelf_mask` | bool | — | подводная континентальная кора не глубже `sea.shelf_max_depth_m` |

Мировые растры над областью модели (область симуляции с ореолом), не зависят от окна:

| Массив | Тип | Смысл |
|---|---|---|
| `world_elevation_m` | float32 `[j, i]` | макровысота на растре каркаса |
| `world_continentality_km` | float32 `[j, i]` | поле края коры на том же растре |

Сетка растра записана в `continents.json → world_grid`: ячейка `(j, i)` имеет центр
`(x0_km + i·cell_km, y0_km + j·cell_km)`.

**Вертикальная система.** Метры; 0 м — нуль модели. Уровень моря — параметр сценария,
`sea_level_m` в `continents.json`. Высота над морем: `max(0, macro_elevation_m − sea_level_m)`,
глубина: `max(0, sea_level_m − macro_elevation_m)`.

**Нумерация.** `continental_block_id` и `landmass_id` — номера связных компонент по общей
стороне гексов в текущем окне, по убыванию площади. `landmass_id` не обязан совпадать с
`continental_block_id`.

## `continents.json` (v2.0)

- `sea_level_m`, `summary` (режим моря, доли суши и шельфа в окне и в области симуляции,
  число крупных массивов, описание структурных операций), `world_grid`, `stage_keys`,
  `warnings`, `validation`.
- `blocks[]`: `id`, `name`, `hexes`, `area_km2`, `centroid_km`, `bounds_km`,
  `touches_window_edge`, `plates` (доли площади по плитам), `plate_count`, `land_area_km2`,
  `submerged_area_km2`, `landmass_ids`.
- `landmasses[]`: те же общие поля и `is_major`, `mean_height_m`, `max_height_m`,
  `on_oceanic_crust_share`, `block_ids`.

Площади и центры относятся к части объекта **внутри окна**; `touches_window_edge` говорит,
что объект продолжается за его краем.

## Данные v3.0

**Контракт эпох.** v1/v2 — неизменяемая исходная эпоха `t = 0`. v3 создаёт новые снимки и
не редактирует массивы v1/v2. `history.epoch` в `config.json` выбирает, какой сохранённый
снимок показан в `history_fields.npz`: `"final"`, `"initial"` или время в млн лет.

**Единицы.** Время — млн лет от исходной эпохи; длина — км; высоты и толщины — м; скорость
плит — мм/год; вертикальная скорость — мм/год. 1 мм/год = 1 км/млн лет.

### `history_fields.npz`

| Слой | Тип | Единицы | Смысл |
|---|---|---|---|
| `elevation_m` | float32 | м | итоговая высота выбранной эпохи; только она задаёт сушу этой эпохи |
| `delta_elevation_m` | float32 | м | `elevation_m` минус `macro_elevation_m` исходной эпохи |
| `plate_id_epoch` | int32 | — | плита, владеющая гексом в эту эпоху |
| `new_crust` | bool | — | кора, созданная за время истории (рифт или новое дно) |
| `boundary_class` | uint8 | — | 0 нет, 1 коллизия, 2 субдукция, 3 расхождение, 4 прочие |
| `continental_fraction` | float32 | 0..1 | доля континентальной коры в эту эпоху |
| `crust_age_ma` | float32 | млн лет | возраст коры |
| `rock_type`, `rock_resistance` | uint8, float32 | —, 0..1 | категория породы (`history.json → summary.rock_types`) и сопротивление эрозии |
| `uplift_rate_mm_yr` | float32 | мм/год | тектоническая вертикальная скорость в эту эпоху |
| `uplift_total_m` | float32 | м | суммарное тектоническое смещение с `t = 0` |
| `erosion_total_m`, `sediment_thickness_m` | float32 | м | снято и отложено с `t = 0`, не меньше нуля |
| `mountain_system_id` | int32 | — | горная система, −1 — нет |
| `mountain_class` | uint8 | — | 0 нет, 1 гребень, 2 склон, 3 перевал, 4 долина, 5 предгорье, 6 вулкан |
| `orogen_age_ma`, `last_tectonic_ma` | float32 | млн лет | время с последнего горообразования и последнего тектонического изменения; NaN — не было |
| `volcanic_potential` | float32 | 0..1 | где магматизм правдоподобен; не вероятность извержения |
| `volcano_id`, `volcanic_age_ma` | int32, float32 | —, млн лет | вулканический центр на гексе и его возраст; −1 и NaN — нет |
| `slope_m_per_km`, `local_relief_m` | float32 | м/км, м | уклон к соседям и перепад непрерывной поверхности внутри гекса |
| `land_mask`, `landmass_id`, `shelf_mask` | bool, int32, bool | — | суша, материки и шельф **выбранной эпохи** |

`base_macro_elevation_m` отдельно не хранится: это `macro_elevation_m` из
`continental_fields.npz`. Одноимённые слои v2 (`land_mask`, `landmass_id`, `shelf_mask`,
`continental_fraction`) остаются в файле v2 и описывают исходную эпоху.

Мировые растры эпохи (префикс `world_`): `elevation_m`, `orogenic_relief_m`, `plate_id`,
`new_crust`, `mountain_system_id`; сетка — `history.json → world_grid`.

### `checkpoints.npz`

Для снимка с номером `i` (времена — `history.json → checkpoint_times_ma`) массивы
`c{i}_{имя}` на растре истории. В материальной системе плит (исходные координаты):
`uplift`, `orogenic`, `erosion`, `sediment`, `volcanic`, `orogen_first`, `orogen_last`,
`orogen_kind`, `last_tectonic`. В системе мантии: `gap_open`, `gap_sediment`.
Положения плит не хранятся: они вычисляются точно из скоростей и времени.

### `events.json`

`id` (устойчивый: `тип:плита-плита:номер`), `type`, `time_ma`, `end_time_ma`, `x_km`, `y_km`,
`plates`, `roles`, `intensity_mm_yr`, `cells`. Типы: `collision_start`, `subduction_start`,
`rifting_start`, `ocean_opening`, `orogeny`, `volcanic_episode`. События с `derived: true`
пересчитываются для выбранной эпохи.

### `mountain_plan.json`

`systems[]` (вид, морфология, возраст, зрелость, острота, площадь, длина), `ridges[]`
(ранг `main` / `branch` / `spur`, родитель, точки в км, относительная высота гребня),
`peaks[]`, `passes[]` (с `between_peaks`), `knots[]`, `valleys[]`. Координаты мировые,
план не обрезается окном.

### Устаревшие данные

При загрузке сверяются версии алгоритмов (`geohistory-1`, `mountains-1`) и ключ стадии
истории. При несовпадении данные v3 **не загружаются**, в `warnings` пишется причина,
исходная эпоха v1/v2 остаётся доступной.

## Данные v4.0

### `hydro_fields.npz`

Слои гексов `[row, col]`. `elevation_m` проекта после v4 — это `post_erosion_elevation_m`;
`pre_erosion_elevation_m` не хранится повторно: это `elevation_m` из `history_fields.npz`.

| Слой | Тип | Единицы | Смысл |
|---|---|---|---|
| `post_erosion_elevation_m` | float32 | m | surface after climate-driven erosion |
| `v4_delta_elevation_m` | float32 | m | post_erosion minus pre_erosion elevation |
| `latitude_deg` | float32 | deg | conditional latitude, north positive |
| `temperature_mean_c` | float32 | degC | annual mean air temperature |
| `temperature_seasonal_c` | float32 | degC | half of the summer-winter range |
| `temperature_djf_c` | float32 | degC | mean air temperature, December-February |
| `temperature_jja_c` | float32 | degC | mean air temperature, June-August |
| `wind_u_ms` | float32 | m/s | prevailing wind, +x east |
| `wind_v_ms` | float32 | m/s | prevailing wind, +y south |
| `precipitation_mm_yr` | float32 | mm/yr | annual precipitation (sum of the seasons) |
| `precipitation_djf_mm` | float32 | mm | precipitation of December-February |
| `precipitation_jja_mm` | float32 | mm | precipitation of June-August |
| `potential_evapotranspiration_mm_yr` | float32 | mm/yr | potential evapotranspiration |
| `actual_evapotranspiration_mm_yr` | float32 | mm/yr | actual evapotranspiration (seasonal soil bucket; open water = PET) |
| `humidity_index` | float32 | P/PET | aridity index, capped at 5 |
| `snow_fraction` | float32 | 0..1 | share of precipitation falling as snow |
| `snow_storage_mm` | float32 | mm | seasonal range of the snowpack, water equivalent |
| `snowmelt_share` | float32 | 0..1 | share of the water supply that comes as melt / ice |
| `glacial_potential` | float32 | 0..1 | climatic potential for perennial ice |
| `infiltration_loss_mm_yr` | float32 | mm/yr | water lost to deep groundwater |
| `runoff_mm_yr` | float32 | mm/yr | runoff to rivers: quick flow + baseflow + glacier discharge (P - AET - deep - dS) |
| `baseflow_index` | float32 | 0..1 | share of the runoff that comes from groundwater |
| `runoff_peak_season` | uint8 | enum | peak runoff season 0 DJF 1 MAM 2 JJA 3 SON, 255 none |
| `flow_receiver` | int64 | index | flat index of the downstream hex (itself = outlet) |
| `flow_direction` | uint8 | enum | 0..5 hex direction to the receiver, 255 = outlet |
| `flow_accumulation_area_km2` | float32 | km2 | upstream land area including the hex |
| `discharge_m3_s` | float32 | m3/s | annual mean discharge leaving the hex |
| `external_inflow_m3_s` | float32 | m3/s | water entering the window from outside |
| `basin_id` | int32 | id | drainage basin, -1 = sea |
| `lake_id` | int32 | id | lake, -1 = none |
| `lake_level_m` | float32 | m | lake water level (NaN = no lake) |
| `lake_depth_m` | float32 | m | lake depth (NaN = no lake) |
| `endorheic_mask` | bool | bool | drains to a closed (terminal) lake |
| `channel_loss_m3_s` | float32 | m3/s | transmission loss of the channel in the hex |
| `outflow_share` | float32 | 0..1 | share of the hex's water reaching sea or window edge |
| `no_outflow_mask` | bool | bool | less than 10 % of the water reaches an outlet |
| `river_mask` | bool | bool | river: discharge, upstream area and system basin reach the hydrology thresholds |
| `river_width_m` | float32 | m | channel width W = 5 Q^0.5 on river hexes, 0 elsewhere |
| `river_depth_m` | float32 | m | channel depth D = 0.3 Q^0.4 on river hexes, 0 elsewhere |
| `strahler_order` | uint8 | order | Strahler order of river hexes, 0 = no river |
| `river_mouth` | bool | bool | last hex of a river before sea, lake or window edge |
| `coastal_discharge_m3_s` | float32 | m3/s | discharge delivered to the sea at mouths |
| `sediment_delivery_m3_yr` | float32 | m3/yr | sediment delivered to the sea at mouths |
| `erosion_rate_m_ma` | float32 | m/Ma | mean v4 erosion rate |
| `v4_erosion_m` | float32 | m | rock and sediment removed during v4 (>= 0) |
| `v4_deposition_m` | float32 | m | material deposited during v4 (>= 0) |
| `sediment_thickness_m` | float32 | m | loose sediment cover after v4 (>= 0) |
| `sediment_flux_m3_yr` | float32 | m3/yr | sediment passing through the hex (last step) |
| `slope_m_per_km` | float32 | m/km | steepest slope to a neighbouring hex |
| `land_mask` | bool | bool | elevation_m > sea_level_m (final surface) |
| `landmass_id` | int32 | id | connected land of the final surface |
| `shelf_mask` | bool | bool | submerged continental crust shallower than the shelf limit |
| `coast_type` | uint8 | enum | coast type of a land hex at the sea, 0 = not coast |
| `wave_exposure` | float32 | 0..1 | open water and storms facing the coast (proxy) |
| `tidal_range_m` | float32 | m | mean tidal range at the coast (proxy) |
| `shelf_width_km` | float32 | km | distance from the coast to water deeper than 200 m |
| `sea_surface_temperature_c` | float32 | degC | mean temperature of the adjacent sea |
| `lgm_ice_mask` | bool | bool | under ice at the last glacial maximum (estimate) |

Сезоны: DJF — декабрь–февраль, JJA — июнь–август (лето северного полушария — JJA,
южного — DJF). Сезонные осадки — в мм за сезон; годовые слои — суммы четырёх сезонов.
`runoff_peak_season`: 0 DJF, 1 MAM, 2 JJA, 3 SON, 255 — стока нет.

`coast_type`: 0 none, 1 ice, 2 fjord, 3 skerry, 4 volcanic, 5 delta, 6 estuary, 7 liman, 8 coral_reef, 9 mangrove, 10 sabkha, 11 tidal_flat, 12 fault_scarp, 13 dalmatian, 14 ria, 15 cliff, 16 barrier_lagoon, 17 beach. Слои факторов берега — NaN вне берега (не ноль).

### `hydrology.json`

- `algorithm_versions` (`climate-3`, `hydro-3`), `units`, `provenance` (какая поверхность
  на входе и какая итоговая), `sea_level_m`, `epoch_time_ma`, `latitude` (широта центра,
  км на градус, признак сжатия, «север — вверх»).
- `summary.world_climate` — бюджет влаги мира за год (мм/год среднего по миру):
  испарение, осадки, приток и отток через границу области, изменение запаса, невязка
  (`relative_residual` ~1e-15), осадки суши по сезонам. `summary.water_balance` — баланс
  суши окна: P, ET, сток, сток ледников, глубокие потери, изменение запасов, наибольшая
  невязка по гексу. `summary.external_inflow` — точки притока извне и выходы воды из окна
  (`exits`, `edge_outlet_hexes`).
- `summary` — средние по суше, доли аридности (UNEP), коэффициент стока, доли озёр,
  бессточных областей, число рек, озёр и сухих котловин, средний размыв; `coasts` — число
  береговых гексов, доли типов, доли скалистых и песчаных берегов.
- `sediment_budget_m3` — снято рекой, перемещено по склонам, отложено на суше, в озёрах,
  в море, ушло в открытое море и за край окна, поднятие, невязка.
- `evolution_steps` — по каждому шагу связки: время, объёмы, наибольший врез, число озёр.
- `river_segments` — сегмент: `id`, `cells` (плоские индексы гексов), `x_km`/`y_km`
  (полилиния), `strahler`, `length_km`, `discharge_m3_s`, `drainage_area_km2`,
  `mean_slope_m_per_km`, `width_m`, `depth_m`, `velocity_m_s` (гидравлическая геометрия
  в последнем гексе), `mouth` (`none` — впадает в следующий сегмент, `sea`, `lake`,
  `endorheic`, `window_edge`), `next_segment`, `basin_id`.
- `river_systems` — от устья к истоку по большему притоку: `mouth_segment`, `main_stem`,
  длина главного русла, водосбор, расход, порядок.
- `lakes` — `cells`, площадь, уровень, наибольшая глубина, `closed` (бессточное), приток,
  сток, гекс перелива. `basins` — площадь и тип окончания (`sea`, `window_edge`,
  `endorheic`).

### Устаревшие данные v4

Версии алгоритмов и ключ стадии климата сверяются при загрузке. Если они не совпадают
(другая эпоха, уровень моря или параметры), данные v4 **не загружаются**, причина пишется в
`warnings`; v1–v3 остаются.

## Данные v4.5 (экологический фундамент)

Файлы: `environment_fields.npz` (поля гексов `[row, col]`), `environment_monthly.npz`
(массивы `(12, rows, cols)`, месяц 0 = январь, float32), `environment.json` (календарь,
методы и происхождение каждого поля, сводка, ключ стадии). NoData — NaN вне суши; маски
`soil_present`, `land_mask` (v4). Нули на месте отсутствующих данных не пишутся.

### `environment_monthly.npz`

| Поле | Единицы | Смысл |
|---|---|---|
| `temperature_monthly_c` | degC | monthly mean air temperature |
| `precip_monthly_mm` | mm/month | precipitation (rain + snow water equivalent) |
| `pet_monthly_mm` | mm/month | reference potential evapotranspiration |
| `aet_monthly_mm` | mm/month | actual evapotranspiration of the reference cover |
| `snowpack_monthly_swe_mm` | mm | snow water equivalent at the end of the month |
| `soil_storage_monthly_mm` | mm | plant-available water at the end of the month |
| `runoff_monthly_mm` | mm/month | local runoff (quick flow + baseflow), not river Q |
| `groundwater_recharge_mm` | mm/month | recharge of the model groundwater store |

### `environment_fields.npz`

| Поле | Тип | Единицы | Смысл |
|---|---|---|---|
| `soil_present` | bool | bool | soil on this land hex (not bare rock, ice or water) |
| `parent_material_class` | uint8 | enum | parent material, stages.environment.PARENT_MATERIALS |
| `sand_frac` | float32 | 0..1 | sand share of the fine earth; NaN where no soil |
| `silt_frac` | float32 | 0..1 | silt share of the fine earth; NaN where no soil |
| `clay_frac` | float32 | 0..1 | clay share of the fine earth; NaN where no soil |
| `coarse_fragment_frac` | float32 | 0..1 | stones by volume |
| `soil_depth_m` | float32 | m | soil thickness (static potential, approximate) |
| `soil_organic_potential` | float32 | 0..1 | organic matter potential (estimate) |
| `field_capacity_vol` | float32 | m3/m3 | FC at -33 kPa, Saxton & Rawls 2006 |
| `wilting_point_vol` | float32 | m3/m3 | PWP at -1500 kPa, Saxton & Rawls 2006 |
| `porosity_vol` | float32 | m3/m3 | saturated water content |
| `awc_mm` | float32 | mm | available water in the reference root zone (0 = no soil) |
| `ks_mm_h` | float32 | mm/h | saturated hydraulic conductivity |
| `infiltration_capacity_mm_day` | float32 | mm/day | daily intake limit |
| `terrain_rocky_fraction` | float32 | 0..1 | bare rock share of the surface |
| `soil_salinity_proxy` | uint8 | 0/1 | potential salinity (dry closed basins, low coasts) |
| `drainage_class` | uint8 | enum | 0 excessively .. 5 very poorly, 255 water |
| `twi` | float32 | ln m | topographic wetness index ln(a / tan b), grid-dependent |
| `twi_anomaly` | float32 | ln | TWI minus that of one hex on a 1 % slope |
| `hand_m` | float32 | m | height above the first drainage cell along the flow path |
| `wetness_potential` | float32 | 0..1 | potential waterlogging (proxy) |
| `floodplain_potential` | float32 | 0..1 | share of the hex a floodplain can cover (potential) |
| `waterlogging_months` | uint8 | months | months with the soil saturated (proxy) |
| `slope_deg` | float32 | deg | mean hex-scale slope |
| `aspect_deg` | float32 | deg | downslope direction, 0 = north, clockwise; NaN on flats |
| `coastal_marine_influence` | float32 | 0..1 | exp(-distance to the sea / 100 km) |
| `lake_shore_influence` | float32 | 0..1 | exp(-distance to a lake / 30 km) |
| `solar_radiation_toa_mj_day` | float32 | MJ/m2/day | annual mean extraterrestrial radiation |
| `day_length_range_h` | float32 | h | longest minus shortest day |
| `local_runoff_annual_mm` | float32 | mm/yr | sum of the monthly local runoff |
| `runoff_difference_vs_v4_mm` | float32 | mm/yr | v4.5 monthly runoff minus v4 runoff |
| `groundwater_recharge_annual_mm` | float32 | mm/yr | recharge of the model groundwater store |
| `groundwater_storage_proxy_mm` | float32 | mm | mean model groundwater store (not a water table) |
| `persistent_snow_accumulation_mm_yr` | float32 | mm/yr | snow left every year (no ice flow) |
| `permanent_ice_mask` | bool | bool | perennial snow accumulation on ground < 60 m/km |
| `environment_feature_flags` | uint16 | bitset | site flags, pipeline_v45.FLAGS |
| `quality_flags` | uint16 | bitset | quality, pipeline_v45.QUALITY |
| `gdd5_proxy_c_day` | float32 | degC day | growing degree days > 5 degC from monthly means |
| `freezing_degree_days` | float32 | degC day | sum of negative monthly means x days |
| `thawing_degree_days` | float32 | degC day | sum of positive monthly means x days |
| `frost_number` | float32 | 0..1 | air frost number (Nelson & Outcalt 1987) |
| `permafrost_potential` | float32 | 0..1 | permafrost potential from the frost number |
| `warmest_month_c` | float32 | degC | warmest monthly mean |
| `coldest_month_c` | float32 | degC | coldest monthly mean |
| `annual_temperature_range_c` | float32 | degC | warmest minus coldest month |
| `cold_month_count` | uint8 | months | months below 0 degC |
| `dry_month_count` | uint8 | months | months with P < 0.5 PET |
| `growing_month_count` | uint8 | months | months >= 5 degC with AET >= 0.5 PET |
| `precip_seasonality_cv` | float32 | % | coefficient of variation of monthly P |
| `aridity_p_over_pet` | float32 | P/PET | annual aridity index from 12 months |
| `aet_over_pet` | float32 | AET/PET | annual water supply of the reference cover |
| `snow_cover_months` | uint8 | months | months ending with > 10 mm of snow |
| `annual_max_swe_mm` | float32 | mm | largest end-of-month snowpack |

Календарь модели: 365 дней, месяцы 31/28/31/30/31/30/31/31/30/31/30/31 (без високосных).
Морозные дни по месячным средним не считаются (`frost_days_exact` = not calculated); GDD,
мерзлота, поймы, засоление — оценки и потенциалы.

## Данные v4.6 (массивы, бассейны, дно, вода)

Файлы `terrain_fields.npz` (слои гексов, `[row, col]`) и `terrain.json` (планы, сводка,
ключи стадии, единицы). Раздел манифеста `v46` хранит `terrain_hash` и хэши массивов.
Проект без них открывается, вкладки v4.6 говорят «не рассчитан».

**Порядок стадий:** v3 (только чтение) → v4.6 terrain (поверхность до эрозии) → v4 (климат,
эрозия, реки, озёра) → v4.6 finish (каньоны и конусы, вода бассейнов, графы массивов,
формы дна) → v4.5.

**Вертикаль.** Метры от нуля модели; уровень моря — параметр сценария.
`surface_elevation_m = bedrock_elevation_m + sediment_thickness_m` (осадок не добавляется
дважды). Итоговая поверхность после эрозии и каньонов — `hydro_fields.npz -> elevation_m`.

| Слой (v4) | Тип | Смысл |
|---|---|---|
| `ocean_mask` | bool | ниже уровня моря **и** связано с мировым океаном |
| `inland_water_mask` | bool | озеро / внутреннее море со своим уровнем |
| `dry_land_mask` | bool | ни то, ни другое; три маски взаимоисключающие |
| `below_sea_level_land_mask` | bool | сухая суша ниже уровня моря |
| `above_sea_level_mask` | bool | только диагностика: `elevation_m > sea_level_m` |
| `lake_surface_level_m`, `water_surface_level_m` | float32, м | уровень воды; NaN на суше |
| `water_depth_m` | float32, м | вода над твёрдой поверхностью, 0 на суше |

`land_mask` = `~ocean_mask` (раньше было `elevation_m > sea_level_m`).

Слои `terrain_fields.npz` перечислены с единицами в `terrain.json -> layers`. Главные:
`surface_elevation_m`, `bedrock_elevation_m`, `oceanic_crust_age_ma`, `basement_depth_m`,
`marine_province` (имена — `terrain.json -> province_names`), `margin_type`,
`fracture_zone_mask`, `trench_axis_mask`, `seamount_type` (1 гора, 2 гайот, 3 остров),
`basin_id` (индекс в `plans.basins`), `basin_subsidence_m`, `basin_fill_m`,
`spill_level_m`, `prominence_m`, `dissection_index`, `submarine_fan_m`, `canyon_depth_m`.

`terrain.json -> plans`:
- `basins` — тип, причина, геометрия (центр, ось, ширины), опускание и доля заполнения, вода (`z_min_m`, `z_spill_m`, `lake_level_m`, сухие гексы ниже моря);
- `massifs` — `ridge_graph` (узлы, рёбра с id хребтов), `valley_graph` (сегменты рек v4 с выходами), вершины с относительной высотой, перевалы, качество;
- `canyons` — путь каньона (км), источник, расход наносов, объём конуса.

Новые слои v3 (`history_fields.npz`): `material_x_km`, `material_y_km`,
`continentality_km`, `base_component_m`, `trench_relief_m`, `trench_distance_km`,
`trench_side`, `trench_rate_mm_yr`, `spreading_axis_distance_km`.

Разрез: `hexworld section PROJECT X0 Y0 X1 Y1 --csv f.csv` → колонки
`distance_km,x_km,y_km,height_m,bedrock_m,water_surface_m,material_type,province`.

## Данные v4.7 (высотная поясность, топоклимат, субгексные условия)

Файлы `topoclimate_fields.npz` (слои гексов `[row, col]`), `topoclimate_monthly.npz`
(`(12, rows, cols)`, 0 = январь), `topoclimate_units.npz` (участки рельефа) и
`topoclimate.json` (единицы, названия поясов, массивы, `provenance`, сводка, ключи).
Раздел манифеста `v47` хранит `topo_hash` и хэши всех массивов. v4.7 загружается только
если версия алгоритма и ключ стадии совпадают с текущими v3 / v4.6 / v4 / v4.5 и
конфигурацией; иначе — предупреждение «NOT loaded (not calculated)».

**Участки рельефа (terrain units, TU)** — статистические страты гекса (форма × высотная
полоса × экспозиция; вода — отдельно), без выдуманного размещения внутри гекса.
Хранение CSR: участки гекса `h = row * width + col` — индексы
`hex_offsets[h] .. hex_offsets[h + 1]`.

| Массив `topoclimate_units.npz` | Форма | Смысл |
|---|---|---|
| `hex_offsets` | (N + 1,) int64 | CSR-индекс |
| `unit_hex` | (U,) int32 | гекс участка |
| `unit_weight_hex`, `unit_weight_land` | (U,) float32 | доля площади гекса / суши гекса |
| `unit_surface_type` | (U,) uint8 | 0 суша, 1 озеро, 2 море (`surface_types`) |
| `unit_landform` | (U,) uint8 | `landforms`: open_slope, ridge, valley_floor, steep_rock |
| `x_km`, `y_km`, `z_surface_m`, `z_ref_m` | (U,) | точка-представитель (реальная точка поверхности) и опорная высота грубого климата |
| `slope_deg`, `aspect_deg` | (U,) | уклон; экспозиция по часовой от севера, NaN на ровном |
| `horizon_rad` | (U, K) float16 | знаковый горизонт по K секторам азимута от севера |
| `svf_slope`, `svf_horizontal`, `visible_sky_fraction`, `upwind_slope_rad` | (U,) | разные метрики неба; Sx против ветра |
| `temperature_c`, `precip_mm`, `pet_mm`, `aet_mm`, `swe_mm`, `runoff_mm`, `radiation_mj_m2_day`, `rh_frac` | (12, U) float32 | месячные поля участка |
| `gsl_t_days`, `gst_t_c`, `gsl_site_days`, `gdd0/5/10`, `fdd`, `thermal_margin`, `thermal_tree_score`, `site_tree_score`, `f_*` | (U,) | сезон роста и пригодность (потенциал, не лес) |
| `thermal_belt` | (U,) uint8 | индекс в `topoclimate.json -> belts` |
| `cap_static`, `cap_jan_c`, `cloud_potential_max`, `fog_potential_mean`, `precip_factor`, `permafrost_potential`, `snow_surplus_mm_yr` | (U,) | топоклимат |

Слои гексов (полный список с единицами — `topoclimate.json -> layers`):
`topo_dry_fraction + topo_lake_fraction + topo_ocean_fraction = 1`; гипсометрия
`hyps_q05/q25/q50/q75/q95_m`, `hyps_z_mean_m`, `hyps_z_std_m`, `hyps_relief_m`;
`belt_share_<пояс>` — доли **суши** гекса (сумма 1 на суше; доля всего гекса = доля ×
`topo_dry_fraction`); `dominant_thermal_belt`; `treeline_proxy_m` и `treeline_state`
(1 граница внутри гекса, 2 весь гекс холоднее, 3 весь теплее); снег, мерзлота,
холодные карманы, облака, туманы. Месячные: `topo_temperature_c`,
`topo_delta_temperature_c`, `topo_radiation_mj_m2_day`, `topo_swe_mm`,
`topo_snow_cover_frac`, `topo_zero_degree_level_m`, `topo_snowline_m`.

Разрез с v4.7 получает колонки `t_mean_c,t_month_c,p_mm_yr,pet_mm_yr,soil_depth_m,
swe_max_mm,gsl_days,gst_c,thermal_margin,belt,quality` (`quality`:
`tu_height_interpolated` — интерполяция по высоте между участками гекса,
`tu_representative`, `tu_height_clamped`, `water`, `unavailable`).

## Данные v5.0 (биомы, покров, криосфера)

Файлы `biomes_fields.npz` (слои гексов `[row, col]`), `biomes_units.npz` (результаты
участков), `biomes_fractions.npz` (доли) и `biomes.json` (реестр классов и PFT, биты причин и
местообитаний, Кёппен, сводка, `provenance`, seed, ключи). Раздел манифеста `v50` хранит
`biome_hash` и хэши массивов.

**Доли** (CSR, по плановой площади всего гекса): классы гекса `h = row * width + col` —
`biome[offsets[h]:offsets[h+1]]`, доли — `fraction[...]`, сумма 1. Доли суши = доля класса /
(1 − `cover_open_water`).

Главные слои: `dominant_biome`, `dominant_land_biome` (коды реестра), `biome_family`,
`realm`, `climate_zone_koppen`, `vegetation_fraction`, `tree_canopy_fraction`, `lai_proxy`,
`productivity_index` (безразмерный), `cover_<класс>` (9 классов покрова, сумма 1),
`pft_<тип>` (22 PFT), `aridity_index` + `aridity_valid`, `moisture_index` (AET/PET),
`dry_season_months`, `growing_months`, `main_limiting_factor`, `limiting_flags`,
`habitat_modifiers` (биты), `ice_thickness_m`, `ice_form` (0 нет, 1 горный ледник, 2 шапка,
3 щит), `ice_id`, `ice_mass_balance_mm`, `firn_swe_mm`, `perennial_snow_fraction`,
`glacier_fraction`, `snow_months`, `ice_initial_condition`, `glacier_state`,
`sea_ice_potential_months`.

Участки (`biomes_units.npz`): `unit_hex`, `unit_weight_hex`, `unit_surface_type`, `biome`,
`limiting_flags`, `habitat_modifiers`, `vegetation_fraction`, `canopy_fraction`,
`productivity_index`, `pft_cover` (22 × U), `cover_fractions` (9 × U), `aridity_index`
(−1 = не определён), `moisture_index`, `dry_season_months`, `z_m`, `cryo_*` (снег, фирн,
возраст и плотность фирна, лёд, толщина, баланс, таяние, снегопад, поток льда).

Отсутствие раздела `v50` означает «не сгенерировано», а не «нулевая растительность».

## Данные v5.2 (плодородие почв и полезные ископаемые)

Файлы `resources_fields.npz`, `resources_index.npz`, `resources.json`; раздел манифеста
`v52` (версия алгоритма `hexworld-resources-5.2.1`, хэши массивов, `stage_keys`). Данные
привязаны к ключу стадии: другая геология, климат или настройки → данные не загружаются
(«не сгенерировано», никогда «нет ресурсов»).

`resources_fields.npz` — плотные слои гексов (`RESOURCE_LAYER_SPECS` в `pipeline_v52.py`):

| Слой | Форма | Единицы | Смысл |
|---|---|---|---|
| `soil_layer_bottom_m`, `soil_bulk_density_kg_m3`, `soil_soc_layers_kg_m2`, `soil_ph_layers` | (3, H, W) | м, кг/м³, кг C/м², pH | функциональные слои 0–0.3 / 0.3–1.0 м / глубже (не генетические горизонты) |
| `soil_soc_kg_m2`, `soil_organic_N_kg_m2`, `soil_ammonium_N_kg_m2`, `soil_nitrate_N_kg_m2` | (H, W) | кг/м² | запасы профиля |
| `soil_P_{primary,labile,sorbed,organic,occluded}_kg_m2`, `soil_K_{mineral,exchange,solution}_kg_m2` | (H, W) | кг элемента/м² | P и K, не P₂O₅ / K₂O |
| `soil_cec_cmolc_kg`, `soil_base_saturation_frac`, `soil_ph`, `soil_acid_equivalent_molc_m2`, `soil_buffer_capacity_molc_m2_per_ph`, `soil_carbonate_kg_m2` | (H, W) | | химия и буферность (pH — proxy) |
| `soil_salt_mass_kg_m2`, `soil_salinity_ec_ds_m`, `soil_esp_pct` | (H, W) | кг/м², dS/m, % | соли (EC — proxy по массе и поровой воде) |
| `soil_fertility_potential`, `suitability_<профиль>`, `soil_limiting_factor` | (H, W) | 0..1, enum | диагностика, не урожайность |
| `prospectivity_by_family`, `resource_in_situ_t_by_family` | (9, H, W) | 0..1, т | группы `resources.models.FAMILIES` |
| `deposit_count`, `shallowest_body_depth_m`, `surface_occurrence_mask`, `overburden_m`, `resource_accessibility_proxy` | (H, W) | | агрегаты каталога |
| `geo_lithology_inferred`, `construction_material_class`, `construction_material_quality` | (H, W) | enum, 0..1 | адаптер литологии (не заменяет `rock_type`), стройматериалы |

NoData: почвенные поля — NaN там, где нет почвы (скала, лёд, вода).

`resources.json` — каталог: `systems` (месторождения: генетический тип, эпоха, происхождение,
руда т, содержания — массовые доли элемента, содержащийся компонент т, сохранность, глубина
кровли, выход на поверхность, `discovery_state`), `bodies` (параметрические тела: форма,
центр и полуоси в м мировой системы / датума, азимут, кровля / подошва, объём, плотность,
доля эрозии), `provinces`, `eroded` (снятые эрозией массы — источник россыпей), сводка с
балансами (`element_budget_over_soil_age`, `erosion_transport_ledger`, `placer_ledger`,
`closed_basins`, `monthly_check`). `resources_index.npz` — CSR «гекс → тела» (`hex_offsets`,
`hex_bodies`).

## Хэши и версии

- `content_hashes` — SHA-256 от типа, формы и байтов каждого массива. Не зависит от
  упаковки файла; это и есть контрольные хэши геологии.
- `geology_hash` — общий хэш всех массивов.
- `file_hashes` — SHA-256 файлов, проверяется при загрузке.
- `stage_keys` — ключ каждой стадии: хэш её параметров и ключей стадий, от которых она зависит.
- `world_id` — первые 16 знаков хэша ключей `plates`, `crust`, `motion`. Одинаков для любых
  окон и разрешений одного мира.
- `geology_hash` и `content_hashes` охватывают только поля v1.0, поэтому не меняются при
  добавлении v2.0. У v2.0 свои `v2.content_hashes` и `v2.geo_hash` в манифесте.

## Программный доступ

```python
from hexworld import Config, generate, save_project, load_project

project = generate(Config(seed="Ab12", width=200, height=150, hex_spacing_km=40.0))
project.layer("uplift")  # массив [row, col]
project.hex_info(col=20, row=30)  # все значения одного гекса
project.sample_world(x_km, y_km)  # геология в произвольных мировых точках
project.sample_region(bounds, hex_spacing_km=5.0)  # тот же мир на другой сетке
save_project(project, "out/p")
load_project("out/p")

# v2.0
from hexworld import generate_v2, derive_landmasses, get_layer

project = generate_v2(load_project("old_v1_project"))  # данные v1 не пересчитываются
project.layer("macro_elevation_m")
flooded = derive_landmasses(project, sea_level_m=150)  # только суша, материки, шельф
layout, elev = get_layer(project, "macro_elevation_m", bounds, hex_spacing_km=5.0)

# v3.0
from hexworld import generate_history, select_epoch, history_events, sample_points

project = generate_history(load_project("old_v2_project"))  # исходная эпоха не меняется
project.layer("elevation_m")
project.layer("base_macro_elevation_m")
middle = select_epoch(project, 15.0)  # без повторной симуляции

# v4.0
from hexworld import generate_v4, generate_climate, build_hydrology, erode

project = generate_v4(project)  # v1-v3 не пересчитываются
project.layer("elevation_m")  # итоговая высота после эрозии
project.stage_layer("v3", "elevation_m")  # поверхность эпохи до эрозии
project.hydro.rivers  # сегменты рек
world, hex_layers = generate_climate(project)  # климат поверхности эпохи v3
history_events(project, time_range=(0, 10), kinds=["collision_start"])
sample_points(project, x_km, y_km, fields=("elevation_m", "rock_type"))
```

Полный пример — `examples/api_usage.py`.
