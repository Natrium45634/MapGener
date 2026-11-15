# PMGE — заметки для работы над кодом

- Сборка и тесты: `cmake -S . -B build && cmake --build build -j && ctest --test-dir build`.
- Публичный контракт — `include/pmge/pmge.h` (C ABI, версия `PMGE_ABI_VERSION`). Менять
  раскладку структур или сигнатуры можно только вместе с увеличением версии ABI и
  адаптером HexWorld (`src/hexworld/mountains/pmge_binding.py` в корне репозитория).
- Никакой глобальной случайности: только `Rng` / `hash_combine` от сида запроса.
- Результат не должен зависеть от числа потоков: каждое значение — чистая функция `(plan, x, y)`.
- Без fast-math. Без нормализации карты целиком: пределы высот — точечно.
- Тесты — `tests/test_pmge.cpp` через C ABI; сценарии интеграции — `examples/scenarios.c`.
