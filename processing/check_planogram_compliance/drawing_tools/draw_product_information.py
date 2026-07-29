import os
import cv2
from dotenv import load_dotenv

from processing.check_planogram_compliance.drawing_tools.draw_text import TextDrawer
drawer = TextDrawer()

load_dotenv()

# Цвет рамки по статусу: match — зелёный, mismatch — жёлтый (не тот товар или лишний),
# нет статуса — чёрный (товар не попал ни в одну линию разметки)
STATUS_COLORS = {
    'match': tuple(map(int, os.getenv('COLOR_MATCH').split(','))),
    'mismatch': tuple(map(int, os.getenv('COLOR_MISMATCH').split(','))),
}
UNUSED_BY_MARKUP_COLOR = tuple(map(int, os.getenv('COLOR_UNUSED').split(',')))
TEXT_NAME_COLOR = tuple(map(int, os.getenv('COLOR_TEXT_NAME').split(',')))
TEXT_CONFIDENCE_COLOR = tuple(map(int, os.getenv('COLOR_TEXT_CONFIDENCE').split(',')))

# Верхний слой: рамка + подпись товара, поверх всего нарисованного ранее, ничего не стирая
def draw_product_information(image, all_products):
    for product in all_products:
        color = STATUS_COLORS.get(product.get('status'), UNUSED_BY_MARKUP_COLOR)

        # 1. Тонкая рамка товара
        cv2.rectangle(image, (product['x1'], product['y1']), (product['x2'], product['y2']), color, 1)

        # 2. Берем всё название после n-го символа
        full_suffix = product['name'][0:].strip()

        # 3. Разбиваем длинный текст на части по 10 символов для переноса
        # Это создаст список строк, например: ['Яблочный', 'Добрый', '0.5л']
        chunk_size = 10
        chunks = [full_suffix[i:i + chunk_size] for i in range(0, len(full_suffix), chunk_size)]

        # Добавляем уверенность отдельной строкой в начало или конец
        chunks.append(f"{int(product['confidence'])}%")

        # 4. Выводим строки одну под другой внутри рамки
        line_height = 12  # Расстояние между строками в пикселях
        for i, line_text in enumerate(chunks):
            # Рассчитываем Y для каждой строки
            # +5 — небольшой отступ от верхнего края рамки
            y_offset = product['y1'] + 5 + (i * line_height)

            # Проверка: если текст уходит ниже дна рамки, перестаем рисовать
            if y_offset > product['y2'] - 5:
                break

            current_color = TEXT_CONFIDENCE_COLOR if "%" in line_text else TEXT_NAME_COLOR

            image = drawer.draw_text(
                image,
                line_text,
                (product['x1'] + 3, y_offset),
                font_size=7,
                color=current_color
            )
    return image