import cv2
import numpy as np


def is_center_in_polygon(point, polygon):
    """Проверяет, находится ли точка (центр товара) внутри полигона."""
    # Преобразуем список точек в формат, который понимает OpenCV (целые числа)
    poly_array = np.array(polygon, dtype=np.int32)
    # Точка должна быть кортежем (x, y)
    pt = (float(point[0]), float(point[1]))

    # pointPolygonTest возвращает:
    # +1 (внутри), 0 (на границе), -1 (снаружи)
    res = cv2.pointPolygonTest(poly_array, pt, False)
    return res >= 0


def get_polygon_area(polygon):
    """Вычисляет площадь полигона в пикселях."""
    poly_array = np.array(polygon, dtype=np.int32)
    return cv2.contourArea(poly_array)


def get_void_percentage(void_area, total_area):
    """Считает процент пустоты."""
    if total_area <= 0: return 0
    return round((void_area / total_area) * 100, 1)