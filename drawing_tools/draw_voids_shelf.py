import cv2
import os
import numpy as np
from dotenv import load_dotenv


class DrawVoidsShelf:
    def __init__(self):
        # Загружаем переменные окружения один раз при создании экземпляра класса
        load_dotenv()

        # Настройки шрифта
        font_str = os.getenv('FONT', 'FONT_HERSHEY_SIMPLEX')
        self.font = getattr(cv2, font_str)
        self.font_scale = float(os.getenv('FONT_SCALE', 0.5))
        self.thickness = int(os.getenv('THICKNESS', 1))

        # Настройки цветов
        self.color_line = self._parse_color('COLOR_LINE')

    def _parse_color(self, env_name):
        """Вспомогательный метод для получения (R,G,B) из .env"""
        color_str = os.getenv(env_name)
        if color_str:
            return tuple(map(int, color_str.split(',')))
        return (255, 255, 255)  # Белый по умолчанию

    def get_void_color(self, percent):
        """Определяет цвет в зависимости от степени пустоты полки"""
        if percent < 10:
            return (0, 255, 0)  # Зеленый (хорошо)
        elif percent < 40:
            return (0, 255, 255)  # Желтый (внимание)
        else:
            return (0, 0, 255)  # Красный (критично)

    def draw(self, image, line_np, mask, percent_void):
        """
        Визуализирует результат анализа на изображении:
        подсвечивает пустоты, рисует линию полки и выводит процент.
        """
        void_color = self.get_void_color(percent_void)

        # 1. Наложение маски (подсветка пустых зон на оригинальном фото)
        # Все пиксели маски, которые > 0, заменяются на цвет void_color
        image[mask > 0] = void_color

        # 2. Отрисовка самой линии полки поверх подсветки
        cv2.polylines(image, [line_np], isClosed=False, color=self.color_line, thickness=1)

        # 3. Подготовка текста
        center_coords = np.mean(line_np, axis=0).astype(int)
        text = f"Void: {percent_void}%"

        # Вычисляем размер текста для центрирования
        (text_w, text_h), _ = cv2.getTextSize(text, self.font, self.font_scale, self.thickness)

        # Координаты для вывода текста (чуть выше центра линии)
        text_x = center_coords[0] - (text_w // 2)
        text_y = center_coords[1] - 15

        # 4. Вывод текста на изображение
        cv2.putText(
            image,
            text,
            (text_x, text_y),
            self.font,
            self.font_scale,
            void_color,
            self.thickness,
            lineType=cv2.LINE_AA  # Сглаживание для красоты
        )

        return image