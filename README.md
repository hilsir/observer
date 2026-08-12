# Observer — автоматическая проверка планограмм

Система по расписанию берёт фото полок, детектирует товары, идентифицирует их и сравнивает выкладку с планограммой из Excel.


> На Fedora с AMD GPU: `HSA_OVERRIDE_GFX_VERSION=11.0.0` выставлен в `main.py` — на других системах закомментировать.

```
observer/
│
├── main.py                                                    # Точка входа. Планировщик по времени (обработка + ежедневный сброс кеша моделей) + run_once() для тестов
├── image_filter.py                                            # Собирает список jpg/png из входной папки (последний снимок на камеру)
│
├── processing/
│   ├── images_processing.py                                   # Оркестратор: цикл по файлам, роутинг → детекция+идентификация → сравнение с планограммой
│   ├── get_models_by_planograms.py                             # Точка доступа к общему экземпляру ModelManager
│   │
│   ├── recognition/
│   │   ├── recognize_products.py                              # Связка: детекция (YOLO) → идентификация вырезанных bbox
│   │   ├── goods_identification.py                             # ResNet50: классифицирует вырезанный bbox → (название, %)
│   │   ├── model_selection.py                                  # ModelManager: кэширует модели идентификации в памяти, отдаёт нужную по ключу модели; reset() сбрасывает кеш
│   │   └── model_processing/
│   │       ├── detector.py                                    # YOLO: детекция товаров, возвращает bbox {x1,y1,x2,y2}
│   │       └── product_identifier.py                           # Вырезает bbox из кадра, прогоняет через модель идентификации
│   │
│   ├── loader/
│   │   ├── load_image.py                                      # Загружает изображение с диска (IMAGES_DIR/group/file)
│   │   ├── load_markup.py                                     # Загружает JSON-разметку полок по имени планограммы (MARKUP_DIR)
│   │   └── load_xlsx.py                                       # Читает Excel-планограмму (PlanogramReader, PLANOGRAMS_DIR)
│   │
│   └── check_planogram_compliance/
│       ├── compare_image_with_palnograms.py                    # По каждой планограмме изображения: подгружает разметку и накладывает сравнение на кадр
│       ├── check_planogram_compliance.py                       # Сравнение с одной планограммой + послойная отрисовка (линии → точки-ориентиры → рамки товаров → % пустоты)
│       ├── planogram_comparator.py                              # Точка доступа к общему экземпляру CompliancePlanogram
│       │
│       ├── checking_planogram/
│       │   ├── compliance_palnogram.py                         # CompliancePlanogram.comparison() — сравнение факта с планограммой по полкам
│       │   ├── get_shelf_segments.py                            # Разбивка разметки полки на сегменты-позиции
│       │   ├── get_shelves/get_shelves.py                       # Группировка линий разметки в полки
│       │   ├── get_planogram_array/                              # Чтение ожидаемого порядка товаров из массива планограммы
│       │   └── comparison/                                       # Хелперы сравнения: пересечение bbox с линией, % пустоты, нормализация имён, мэтчинг сегментов
│       │
│       └── drawing_tools/
│           ├── draw_markup_lines.py                             # Нижний слой: тонкая линия разметки полки
│           ├── draw_shelf_segments.py                            # Точки-ориентиры ожидаемых позиций товаров
│           ├── draw_product_information.py                      # Рамка + подпись товара (цвет по статусу match/mismatch/unused)
│           ├── draw_void_percent.py                              # Цифра % пустоты полки (зелёный/жёлтый/красный)
│           └── draw_text.py                                     # Рендер текста через PIL (кириллица, TTF)
│
├── router/
│   ├── router_models_config.py                                 # get_model_key_for_planogram — читает data_router.json ("models") на каждый вызов
│   └── router_planograms_config.py                              # get_planograms_for_image — читает data_router.json ("images") на каждый вызов
│
├── save_result/
│   └── send_files.py                                            # Сохраняет аннотированное фото и текстовый отчёт (save_locally, RETURN_FILES)
│
├── data_for_processing/
│   ├── input_img/                                                # Входные фото полок (по группам-подпапкам)
│   ├── markup/                                                   # JSON-разметки линий полок
│   ├── planograms/                                               # Excel-планограммы
│   ├── data_router/data_router.json                              # Единый файл роутинга: модель↔планограммы, изображение↔планограммы
│   ├── img_return/                                               # Результаты: фото + отчёт
│   └── marker/                                                   # Утилиты для ручной разметки полок
│
├── models/                                                       # Веса нейросетей (.pt — YOLO, .pth — ResNet50)
└── fonts/                                                        # TTF-шрифты для отрисовки текста
```

## Пайплайн одного изображения

1. `load_image` — загрузка кадра (`IMAGES_DIR`).
2. `router_planograms_config.get_planograms_for_image` — какие планограммы относятся к этому фото (из `data_router.json`).
3. `get_models_by_planograms` → `ModelManager.get_model_for` — по имени планограммы через `router_models_config` находим ключ модели и берём (или лениво загружаем и кешируем) модель идентификации.
4. `recognize_products.get_recognized_goods`:
   - `detector.detect` — YOLO находит все товары на кадре (bbox, без названий);
   - `product_identifier.identify` — каждый bbox вырезается и прогоняется через выбранную модель идентификации (название + уверенность).
5. `compare_image_with_palnograms` — по каждой планограмме этого изображения:
   - `load_markup` — своя JSON-разметка полок;
   - `check_planogram_compliance` → `CompliancePlanogram.comparison()` (сравнение по полкам: совпадения/несовпадения/отсутствующие товары), затем послойная отрисовка на кадр.
6. `save_locally` — сохранение аннотированного фото и текстового отчёта в `RETURN_FILES`.

## Кеширование и обновление данных

- Разметки (`load_markup`) и планограммы (`load_xlsx`) читаются с диска на каждой обработке — изменения подхватываются сразу, без перезапуска.
- `data_router.json` (роутинг изображение↔планограммы, планограмма↔модель) тоже читается заново на каждый вызов — правки подхватываются мгновенно.
- Модели идентификации кешируются в памяти (`ModelManager._models`) на весь срок жизни процесса и сбрасываются раз в сутки по времени `MODELS_RELOAD_TIME` (`.env`) — так переобученные/заменённые модели подхватываются без остановки `main.py`.
- Детектор товаров (YOLO) грузится один раз при старте процесса и не перезагружается — замена `MODEL_GOODS_PATH` требует рестарта.

## Текущее состояние

- Роутинг ведётся через `data_for_processing/data_router/data_router.json` (а не хардкодом в `router_*_config.py`, как раньше); активна модель **10104450** для планограммы «Кофе растворимый_3».
- Отправка в Telegram/Max отключена, результаты только сохраняются локально (`save_locally`).
