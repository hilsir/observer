import cv2
import os
import json
from string_formation.create_path import CreatePath


def load_image(filename):
    # путь к картинке
    image_path = CreatePath.create_path_image(filename)
    # Загружаем изображение в формате BGR (Для OpenCV)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Пропуск нечитаемого изображения: {filename}")
        return None

    return image


def load_markup(filename):
    # путь к разметке
    markup_path = CreatePath.create_path_markup(filename)

    if not os.path.exists(markup_path):
        print(f"Пропуск нечитаемой разметки: {filename} нет JSON")
        return None, None

    # Лист массивов координат выделеных областей
    with open(markup_path, 'r') as f:
        markup = json.load(f)

    return markup, markup_path
