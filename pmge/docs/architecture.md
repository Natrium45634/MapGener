# Архитектура PMGE 0.1

```text
pmge_request ──► validate ──► MountainPlan ──► HeightFieldEvaluator H(x, y) ──► Samplers
                               (plan.cpp)        (sampler.cpp)                   точки, пакеты,
                                                                                 растры, гексы,
                                                                                 патчи (hex.cpp)
                                                       raster ──► Erosion (erosion.cpp)
C ABI (c_api.cpp): opaque pmge_plan*, коды ошибок, pmge_last_error(), без исключений наружу
```

| Уровень ТЗ (п. 6) | Где | Состояние |
|---|---|---|
| MountainPlan | `src/plan.cpp`, `pmge::Plan` | есть: хребты (главный, ветви, отроги двух порядков), вершины со стабильными ID, перевалы со ссылками на вершины, долины |
| HeightFieldEvaluator | `Plan::sample` | есть: непрерывная `H(x, y)`; пространственный индекс по ячейкам |
| Rasterizer / Samplers | `generate_heightmap`, `sample_batch`, `sample_hexes`, `generate_patch` | есть |
| Erosion / Hydrology | `src/erosion.cpp` | термическая и по стоку на растре; гидрологии (рек, бассейнов) нет |
| ConstraintSolver | — | частично: абсолютные пределы, политика constrained_fit; якорей, защищённых областей, граничных уклонов нет |
| Integration / Composition | `generate_patch` | replace / add / max / masked / constrained_fit; патч всегда дельта |
| Backends | CPU | GPU нет |

## Модель высоты (автономный план)

```text
H = base + (max − base) · ceil( a·E + b·P + c·D ) · mask
```

- `E` — тело гор: гладкий купол вокруг главной оси и ветвей шириной «отрог + полгребня»,
  с плавным переходом в предгорья; долины его прорезают.
- `P` — структура: максимум по хребтам `crest · (1 − d/hw)^sharpness`, поверх — формы
  вершин по типу; долины понижают склоны, но не гребень.
- `D` — ограниченная мелкая деталь (ridged-шум), только внутри гор.
- `a, b, c` зависят от зрелости: молодые — больше `P` и `D`, старые — больше `E`.
- `ceil` — мягкий точечный потолок у `max_m`; `mask` — плавный край области.

Высота в любой точке не зависит от того, какие ещё точки запрошены, — отсюда отсутствие швов
и независимость от порядка и числа потоков.

## Внешний план

`pmge_plan_create_external` принимает готовые оси хребтов, долины и сетку амплитуды
(рельефа, который должны дать горы). Результат — дельта `amplitude · gain · detail ·
(P + D − mean_offset)`. Так геологический генератор (HexWorld) задаёт, *где* и *насколько*
поднялись горы, а PMGE — *как* они выглядят.
