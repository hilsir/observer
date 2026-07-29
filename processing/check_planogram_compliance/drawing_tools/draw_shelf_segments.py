import os
import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()
DOT_COLOR = tuple(map(int, os.getenv('COLOR_DOT').split(',')))  # точка — ориентир ожидаемой позиции товара
DOT_RADIUS = 4  # px — маленькая точка, не должна перекрывать содержимое рамки товара


# Y в точке X вдоль изогнутой линии полки (интерполяция) — точка следует изгибу,
# а не сидит на одной высоте по всей ширине
def interpolate_y(line_np, x):
    xs = line_np[:, 0]
    ys = line_np[:, 1]
    order = np.argsort(xs)
    return int(np.interp(x, xs[order], ys[order]))


# Средний слой: красная точка-ориентир в центре каждой ожидаемой позиции товара
class DrawShelfSegments:
    def draw(self, image, shelf_results, markup):
        for shelf, line in zip(shelf_results, markup):
            line_np = np.array(line, np.int32)

            for segment in shelf['segments']:
                center_x = (segment['x1'] + segment['x2']) / 2
                center_y = interpolate_y(line_np, center_x)
                cv2.circle(image, (int(center_x), center_y), DOT_RADIUS, DOT_COLOR, thickness=-1)

        return image
