# qimg — Qwen-Image-2.1 на бесплатных 2×T4 Kaggle

Генерация картинок на видеокартах Kaggle (30 ч GPU в неделю). Что умеет: картинка по тексту
(черновик ×2 / стандарт / максимум), дорисовка выбранного кадра, ×4, правка картинки словами
(до 3 картинок на вход), «похожие» и прозрачный фон. Управление через CLI или локальную страницу.

```
клиент (CLI / страница 127.0.0.1:7861) ──https+Bearer──> *.trycloudflare.com ──> GPU-ядро Kaggle
                       ▲                                                          imgd :8080
                       └──── маяк ntfy.sh/<топик>  (start … ready&url=… shutdown)    ├ sd-server cuda0
                             stop → ntfy.sh/<топик>-ctl                              ├ sd-server cuda1
                                                                                     └ переписчик (GPU1 по времени)
```

## Что нужно

* Аккаунт Kaggle с **подтверждённым телефоном** (без него GPU не дают).
* API-токен: kaggle.com → аватар → **Settings** → **API** → **Create New Token**. Подойдёт строка `KGAT_...`
  или файл `kaggle.json`.
* Python 3.10+, `pip install kaggle pillow` (Pillow нужен только для тестов).

## Первый запуск

```bash
cd qimg
python3 -m venv .venv && . .venv/bin/activate && pip install kaggle
python3 qimg.py init --user <ник> --token KGAT_...     # или --kaggle-json ~/Downloads/kaggle.json
python3 qimg.py stub          # канал на CPU-заглушке: туннель + маяк + stop, GPU-квота не тратится
python3 qimg.py setup         # сборка sd.cpp (~36 мин) + качалки весов (~14.5 ГБ и ~5.5 ГБ), всё на CPU
python3 qimg.py kstatus       # дождаться complete у build / fetch-serve / fetch-pe
python3 qimg.py up            # GPU-ядро: 5–7 мин холодного старта (первый раз до 13)
```

`init` сам генерирует ключ API и топик ntfy и кладёт их в `config.json`. Токен Kaggle уходит в `secrets/`.
Оба пути в `.gitignore`.

## Работа

```bash
python3 qimg.py gen "кот в космосе"                       # черновик: 2 кадра, ~60 с
python3 qimg.py gen "кот в космосе" -p std -a 16:9        # стандарт, ~72 с
python3 qimg.py gen "старый маяк в шторм" -p hq           # максимум, ~270 с, png
python3 qimg.py gen "стеклянная бутылка" --alpha          # прозрачный фон
python3 qimg.py gen "сделай вечер" -i out/…/кадр.webp     # правка по картинке (до 3 -i)
python3 qimg.py refine out/2026-10-03/…webp               # дорисовать кадр
python3 qimg.py vary  out/…    |  up4x out/…  |  rmbg out/…
python3 qimg.py edit a.png b.png "посади собаку с первой картинки на диван со второй"
python3 qimg.py status        # состояние ядра и сколько квоты сожжено за неделю
python3 qimg.py down          # погасить (иначе само гаснет через 15 мин без заказов, потолок 11.5 ч)
python3 qimg.py ui            # страница: http://127.0.0.1:7861
```

Готовые кадры сразу сохраняются в `out/ДАТА/` вместе с JSON (промпт, пресет, сид, размер). Кадр
прошлой сессии можно снова отдать на правку: клиент дозагрузит его через `/v1/upload`.

Короткие запросы на draft/std/hq переписчик Qwen-Image-2.1-PE разворачивает в подробный английский абзац.
На группу это добавляет около 50 с. Отключить можно флагом `--no-rewrite` или галочкой на странице.

### Страница в launchd (мак)

`~/Library/LaunchAgents/local.qimg.plist`: `ProgramArguments` = `[…/.venv/bin/python, …/qimg/qimg.py, ui]`,
`KeepAlive` = true, затем `launchctl load ~/Library/LaunchAgents/local.qimg.plist`.

## Устройство

| файл | что |
|---|---|
| `qimg.py` | клиент: сборка ядер из шаблонов, push, маяк, квота, CLI, локальная страница |
| `presets.json` | пресеты и размеры — одна правда для ядра и клиента |
| `lib/transport.py` | маяк ntfy, сигнал stop, туннель cloudflared, сторож |
| `lib/sdserver.py` | sd-server на одну карту, задания, прогресс по логу |
| `lib/imgd.py` | очередь с приоритетами + HTTP API |
| `lib/rewriter.py` | переписчик промпта (llama-server на GPU1 по времени) |
| `kernels/*.py` | шаблоны ядер; `__X_LIB__` и ключи подставляются на пуше |
| `ui/index.html` | страница |
| `tests/` | локальный прогон на поддельных sd-server/llama-server: `python3 tests/test_local.py` |

Kaggle принимает ядро одним файлом, поэтому `render` склеивает шаблон с модулями из `lib/`.
Посмотреть результат можно так: `python3 qimg.py render serve` → `build/serve/kernel.py`.

## Грабли (из инструкции, учтены в коде)

* `LD_LIBRARY_PATH` только дописывается, иначе sd-server не найдёт Kaggle-овский `libcuda.so.1`.
* `--rng cpu`: с `cuda` правка на части сидов отдаёт пережаренную копию исходника.
* Плитки VAE 256 включаются для всего крупнее 1 Мп, для img2img и hires. Правка идёт без плиток
  (с прозрачностью они дают пустой кадр), поэтому её размер ≤ 1024² и округляется кратно 32 вниз.
* sd-server гасится только своим pid, без `pkill`.
* **Пуш новой версии ядра не отменяет идущую сессию**: `up` откажется, пока старая жива. Сначала `down`.
* GPU-гнездо одно: два GPU-ядра разом не запускать.
* Событие маяка ищется только после своего запуска.
* Неделя квоты считается с субботы 00:00 UTC (`quota_week_start_weekday` в `config.json`, 0 = пн).

## Правила Kaggle

Ноутбуки Kaggle — не постоянный хостинг: за ядро, которое сутками гоняет трафик, аккаунт забанят.
Поднимайте ядро под работу и гасите после. Самогашение не убирайте.
