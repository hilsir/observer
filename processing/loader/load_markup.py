import os
import json

markup_dir = os.getenv('MARKUP_DIR')

def load_markup(name):
    # путь к JSON-файлу по имени планограммы (без расширения)
    markup_path = os.path.join(markup_dir, f"{name}.json")

    if not os.path.exists(markup_path):
        print(f"Пропуск нечитаемой разметки: {name}.json не найден")
        return None, None

    # Лист массивов координат выделеных областей
    with open(markup_path, 'r') as f:
        markup = json.load(f)

    return markup, markup_path
