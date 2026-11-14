# Формат проекта HexWorld (`hexworld-project-2`)

Читаются обе схемы: `hexworld-project-1` (только v1.0) и `hexworld-project-2`.

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
| `config.v1.json`, `manifest.v1.json` | копии исходных файлов, если проект v1.0 был дополнен v2.0 |

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
```

Полный пример — `examples/api_usage.py`.
