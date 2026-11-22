# Установщик HexWorld для Windows

Пользователю нужен только файл `HexWorld-1.0.0-Setup.exe`. Python, Qt, numpy, scipy и
библиотека гор `pmge.dll` уже внутри программы.

## Где взять

- **Релиз.** Тег `v1.0.0` (любой `v*`) запускает workflow
  `.github/workflows/windows-installer.yml`. Он прикрепляет установщик к релизу GitHub.
- **Любая сборка ветки.** На вкладке Actions откройте `windows-installer`, затем
  артефакт «HexWorld-Windows-Setup».

## Как собирается (то же самое делает CI на `windows-latest`)

```powershell
cmake -S pmge -B pmge/build
cmake --build pmge/build --config Release
python -m pip install ".[gui]" pyinstaller
python packaging/windows/make_icon.py packaging/windows/build
pyinstaller packaging/windows/hexworld.spec --noconfirm
& "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" /DAppVersion=1.0.0 packaging\windows\HexWorld.iss
```

Результат — `packaging/windows/Output/HexWorld-1.0.0-Setup.exe`.

| Файл | Что делает |
|---|---|
| `packaging/windows/hexworld_app.py` | точка входа программы. `HexWorld.exe` открывает окно; `HexWorld.exe ПАПКА` открывает сохранённый мир; `--self-test ОТЧЁТ.json` генерирует мир без окна, экспортирует его, открывает главное окно offscreen и выходит с кодом 0 или 1 |
| `packaging/windows/make_icon.py` | значок `hexworld.ico` (16–256 px) и свойства файла `HexWorld.exe` (версия 1.0.0) |
| `packaging/windows/hexworld.spec` | PyInstaller: одна папка, без консоли, рядом лежит `pmge.dll`; ненужные модули Qt исключены |
| `packaging/windows/HexWorld.iss` | Inno Setup: русский и английский; установка для пользователя без прав администратора (или для всех — по выбору); ярлыки в «Пуск» и на рабочем столе; удаление через «Приложения» |
| `packaging/windows/README_WINDOWS.txt` | «Как пользоваться» — ставится рядом с программой и в меню «Пуск» |

## Проверки в CI

1. Собранная папка: `HexWorld.exe --self-test` (генерация до биомов, экспорт для игр,
   главное окно, панель гекса).
2. Тихая установка: `Setup.exe /VERYSILENT /CURRENTUSER /DIR=...`.
3. `--self-test` установленной программы.
4. Тихое удаление; проверяется, что `HexWorld.exe` удалён.

## Где программа хранит данные

- Программа: `%LOCALAPPDATA%\Programs\HexWorld` (для пользователя) или
  `C:\Program Files\HexWorld` (для всех).
- «Мои миры» (история, избранное, миниатюры): `%APPDATA%\HexWorld\library`.
- Замки настроек: реестр, раздел `HexWorld` (QSettings).

## Версии

- `PRODUCT_VERSION` (`hexworld/version.py`) — версия программы: 1.0.0. Её показывают
  установщик, заголовок окна и «О программе».
- `GENERATOR_VERSION` — версия алгоритмов и формата данных (5.2.0). Она записывается в
  сохранённые проекты и продолжает внутреннюю историю генератора (1.0 … 5.1).

## Ограничения

- Размер установленной программы — около 300 МБ, в основном Qt и scipy. Установщик сжат.
- Подписи кода нет, поэтому Windows SmartScreen при первом запуске может спросить
  «Запустить всё равно?». Подпись требует сертификата издателя.
- **NOT VERIFIED:** окно программы на физическом экране Windows. CI проверяет установку
  и запуск в offscreen-режиме, но не реальный экран.
