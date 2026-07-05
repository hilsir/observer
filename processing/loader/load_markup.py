import os
import json

markup_dir = os.getenv('MARKUP_DIR')


def load_markup(filename):
    # Убрать расширение из названия
    filename_no_expansion = os.path.splitext(filename)[0]
    # путь к JSON-файлу с разметкой
    markup_path = os.path.join(markup_dir, f"{filename_no_expansion}.json")

    if not os.path.exists(markup_path):
        print(f"Пропуск нечитаемой разметки: {filename} нет JSON")
        return None, None

    # Лист массивов координат выделеных областей
    with open(markup_path, 'r') as f:
        markup = json.load(f)

    return markup, markup_path
