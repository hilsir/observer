import cv2

from processing.check_planogram_compliance.drawing_tools.draw_text import TextDrawer
drawer = TextDrawer()

def draw_product_information(image, all_products):
    for p in all_products:
        # 1. Тонкая рамка товара
        cv2.rectangle(image, (p['x1'], p['y1']), (p['x2'], p['y2']), (0, 255, 0), 1)

        # 2. Берем всё название после n-го символа
        full_suffix = p['name'][0:].strip()

        # 3. Разбиваем длинный текст на части по 10 символов для переноса
        # Это создаст список строк, например: ['Яблочный', 'Добрый', '0.5л']
        chunk_size = 10
        chunks = [full_suffix[i:i + chunk_size] for i in range(0, len(full_suffix), chunk_size)]

        # Добавляем уверенность отдельной строкой в начало или конец
        chunks.append(f"{int(p['confidence'])}%")

        # 4. Выводим строки одну под другой внутри рамки
        line_height = 12  # Расстояние между строками в пикселях
        for i, line_text in enumerate(chunks):
            # Рассчитываем Y для каждой строки
            # +5 — небольшой отступ от верхнего края рамки
            y_offset = p['y1'] + 5 + (i * line_height)

            # Проверка: если текст уходит ниже дна рамки, перестаем рисовать
            if y_offset > p['y2'] - 5:
                break

            # Цвет для уверенности (последняя строка) сделаем желтым
            current_color = (0, 255, 255) if "%" in line_text else (255, 255, 255)

            image = drawer.draw_text(
                image,
                line_text,
                (p['x1'] + 3, y_offset),
                font_size=7,
                color=current_color
            )
    return image