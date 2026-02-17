import cv2
import os
from dotenv import load_dotenv
from old import geometry_utils as geo

# Загружаем переменные из .env
load_dotenv()

# Вспомогательная функция для получения цвета (как мы делали ранее)
def get_color(env_name, default=(0, 0, 255)):
    color_str = os.getenv(env_name)
    if color_str:
        return tuple(map(int, color_str.split(',')))
    return default

# Считываем константы
SIDE_GAP_LIMIT = int(os.getenv('SIDE_GAP_LIMIT', 50))
COLOR_VOID = get_color('COLOR_VOID')

def find_voids_in_area(image, polygon, products):
    """
    Ищет пустоты только внутри конкретного полигона.
    """
    # 1. Оставляем только те товары, чей центр внутри текущей области
    area_products = [p for p in products if geo.is_center_in_polygon((p['cx'], p['cy']), polygon)]

    # Сортируем товары слева направо
    area_products.sort(key=lambda p: p['x1'])

    total_void_area = 0

    # 2. Анализируем промежутки между товарами
    for i in range(len(area_products)):
        curr = area_products[i]

        # Точка для проверки "пустоты" справа от товара
        check_x = curr['x2'] + SIDE_GAP_LIMIT
        check_y = curr['cy']

        # Если точка впереди всё еще в полигоне
        if geo.is_center_in_polygon((check_x, check_y), polygon):
            has_neighbor = False
            next_product_x = None

            for j in range(i + 1, len(area_products)):
                if area_products[j]['x1'] < check_x + 10:
                    has_neighbor = True
                    break
                else:
                    next_product_x = area_products[j]['x1']
                    break

            if not has_neighbor:
                v_x1 = curr['x2']
                # Если следующего товара нет, ограничиваем визуальную область
                v_x2 = next_product_x if next_product_x else (curr['x2'] + SIDE_GAP_LIMIT * 2)
                v_y1 = curr['y1']
                v_y2 = curr['y2']

                # Рисуем пустоту (полупрозрачный цвет из .env)
                overlay = image.copy()
                cv2.rectangle(overlay, (v_x1, v_y1), (int(v_x2), v_y2), COLOR_VOID, -1)
                cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

                total_void_area += (v_x2 - v_x1) * (v_y2 - v_y1)

    return total_void_area