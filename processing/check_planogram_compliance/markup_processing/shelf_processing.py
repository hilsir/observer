import cv2
import numpy as np
from processing.check_planogram_compliance.drawing_tools.draw_voids_shelf import DrawVoidsShelf

class ShelfProcessing:

    def __init__(self):
        self.draw_voids_shelf = DrawVoidsShelf()

    def process_shelves(self, image, markup, all_products_identified):
        """
        Основной метод обработки линий полок.
        Возвращает модифицированное изображение и максимальный процент пустоты.
        """
        max_percent_void = 0
        image_height, image_width = image.shape[:2]

        for line in markup:
            # Пустая маска из 0
            mask = np.zeros((image_height, image_width), dtype=np.uint8)

            # Преобразуем координаты линии из JSON-разметки в массив NumPy для OpenCV.
            line_np = np.array(line, np.int32)

            # Рисуем линию полки на маске белым цветом (255)
            cv2.polylines(mask, [line_np], isClosed=False, color=255, thickness=5)

            # Считаем количество белых пикселей (площадь линии)
            total_line_pixels = cv2.countNonZero(mask)

            # Если линия не была нарисована (например, пустые координаты), переходим к следующей.
            if total_line_pixels == 0:
                continue

            # Вычитание товаров (обнаруженных объектов) из маски
            for prod in all_products_identified:
                # Извлекаем координаты точек товара и приводим к целым числам
                x1, y1, x2, y2 = map(int, [prod['x1'], prod['y1'], prod['x2'], prod['y2']])

                # Удаляем прямоугольники товара из линии - закрашиваем (0)
                # thickness=-1 означает полную заливку фигуры.
                cv2.rectangle(mask, (x1, y1), (x2, y2), 0, thickness=-1)

            # Считаем оставшиеся белыми пиксели линии на саске
            remaining_pixels = cv2.countNonZero(mask)

            # Площади оставшейся линии и переводим в проценты
            percent_void = round((remaining_pixels / total_line_pixels) * 100, 1)

            # Обновляем глобальный максимум пустоты для вей полки
            if percent_void > max_percent_void:
                max_percent_void = percent_void

            # Отрисоыка пустот на кадре
            image = self.draw_voids_shelf.draw(image, line_np, mask, percent_void)

        return image, max_percent_void


