import cv2
import os
import json
import numpy as np
from detector import ProductDetector
from dotenv import load_dotenv
load_dotenv()

def get_color(env_name):
    """Превращает строку 'R,G,B' из .env в кортеж (R, G, B)"""
    return tuple(map(int, os.getenv(env_name).split(',')))

def image_processing(image_filenames):
    # Берем настройки напрямую из env
    model_goods_path = os.getenv('MODEL_GOODS_PATH')
    images_dir = os.getenv('IMAGES_DIR')
    markup_dir = os.getenv('MARKUP_DIR')

    conf_threshold = float(os.getenv('CONF_THRESHOLD'))
    color_goods = get_color('COLOR_GOODS')
    color_line = get_color('COLOR_LINE')
    percentage_for_notification = int(os.getenv('PERCENTAGE_FOR_NOTIFICATION'))

    # Тип шрифта
    font_str = os.getenv('FONT')
    font_scale_str = os.getenv('FONT_SCALE')
    thickness_str = os.getenv('THICKNESS')

    font = getattr(cv2, font_str)  # Получаем атрибут из модуля cv2 по имени строки
    font_scale = float(font_scale_str)
    thickness = int(thickness_str)

    detector = ProductDetector(model_goods_path, conf_threshold)
    finished_images = []

    print(f"Найдено изображений для обработки: {len(image_filenames)}")

    for filename in image_filenames:
        # Собираем полный путь к картинке (папка + имя файла)
        img_path = os.path.join(images_dir, filename)

        # Загружаем изображение в формате BGR (Для OpenCV)
        image = cv2.imread(img_path)

        # Пропустить нечитаемый файл
        if image is None:
            continue

        # Извлекаем ID камеры из названия
        camera_id = filename.split('_')[0]

        # Путь к JSON-файлу с разметкой
        markup_path = os.path.join(markup_dir, f"{camera_id}.json")

        # Проверяем, есть ли разметка для данной камеры
        if not os.path.exists(markup_path):
            print(f"Пропуск {filename}: нет JSON")
            continue

        # Лист массивов координат выделеных областей
        with open(markup_path, 'r') as f:
            markup = json.load(f)

        # Поиск товаров
        all_products = detector.detect(image)

        # Выделение найденых товаров
        for p in all_products:
            cv2.rectangle(image, (p['x1'], p['y1']), (p['x2'], p['y2']), color_goods, 1)

        max_percent_void = 0

        # Обработка линий полок
        for line in markup:
            # Высота и ширина исходного изображения
            image_height, image_width = image.shape[:2]

            # Массив нулей в 8-bit (маска для разметок)
            mask = np.zeros((image_height, image_width), dtype=np.uint8)

            # Переводим лист с разметкой в массив NumPy для чтения в OpenCV
            line_np = np.array(line, np.int32)

            # Переносим линию на маску
            cv2.polylines(mask, [line_np], isClosed=False, color=255, thickness=10)

            # Считаем все закрашеные пиксели на маске
            total_line_pixels = cv2.countNonZero(mask)

            # Если маска вдруг оказалась пустой, пропускаем её
            if total_line_pixels == 0:
                continue

            # Удаление товаров из линии
            for coordinates_item in all_products:
                # Координаты в целые числа
                x1, y1, x2, y2 = int(coordinates_item['x1']), int(coordinates_item['y1']), int(coordinates_item['x2']), int(coordinates_item['y2'])

                # Где товар соприкосается с линией, пиксели становятся = 0
                # (thickness=-1 закрасить прямоугольник полностью)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 0, thickness=-1)

            # Считаем оставшиеся закрашеные пиксели на маске
            remaining_line_pixels = cv2.countNonZero(mask)

            # percent_void процент пустот на линии
            percent_void = round((remaining_line_pixels / total_line_pixels) * 100, 1)

            # Сохраняем савый большой пропуск
            if max_percent_void < percent_void:
                max_percent_void = percent_void

            # Определяем цвет в зависимости от процента пустоты
            if percent_void < 10:
                void_color = (0, 255, 0)  # Зеленый (все отлично)
            elif percent_void < 40:
                void_color = (0, 255, 255)  # Желтый (пора проверить)
            else:
                void_color = (0, 0, 255)  # Красный (критично пустая полка)

            # Накладываем маску на фото
            # image[line_mask > 0] выбирает все ненулевые пиксели пиксели маски
            image[mask > 0] = void_color

            # Рисуем разметки полки
            cv2.polylines(image, [line_np], isClosed=False, color=color_line, thickness=1)

            # Находим геометрический центр линии (среднее по всем точкам X и Y)
            center_coords_line = np.mean(line_np, axis=0).astype(int)
            center_line_x, center_line_y = center_coords_line[0], center_coords_line[1]

            text = f"Void: {percent_void}%"

            # Вычисляем размер текстового блока для выравнивания
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            # Координаты трекста:
            text_x = center_line_x - (text_width // 2)
            text_y = center_line_y - 15

            # Печатаем текст
            cv2.putText(image, text, (text_x, text_y), font, font_scale, void_color, thickness)

        # Скипаем изображение если не привышает минимальный пропуск на полках
        # if percentage_for_notification >= max_percent_void:
        #     continue

        # Сохраняем результат (имя файла и обработанный кадр) в итоговый список
        finished_images.append((filename, image))

    return finished_images