import os
import numpy as np
from dotenv import load_dotenv

from processing.check_planogram_compliance.drawing_tools.draw_text import TextDrawer
drawer = TextDrawer()

load_dotenv()

FONT_SIZE = 15  # крупнее, чтобы было легко разглядеть
Y_OFFSET_ABOVE_LINE = 30  # px — выше линии полки, чтобы не залезать на подписи товаров над ней

VOID_LOW_COLOR = tuple(map(int, os.getenv('COLOR_VOID_LOW').split(',')))
VOID_MID_COLOR = tuple(map(int, os.getenv('COLOR_VOID_MID').split(',')))
VOID_HIGH_COLOR = tuple(map(int, os.getenv('COLOR_VOID_HIGH').split(',')))


# Цвет цифры процента пустоты: зелёный/жёлтый/красный по степени пустоты полки
def get_void_percent_color(percent_void):
    if percent_void < 10:
        return VOID_LOW_COLOR
    elif percent_void < 40:
        return VOID_MID_COLOR
    else:
        return VOID_HIGH_COLOR


# Самый верхний слой: только цифра % пустоты полки, поверх всего остального
class DrawVoidPercent:
    def draw(self, image, shelf_results, markup):
        for shelf, line in zip(shelf_results, markup):
            segments = shelf['segments']
            if not segments:
                continue

            line_np = np.array(line, np.int32)
            y_min = int(np.min(line_np[:, 1]))

            percent_void = shelf['percent_void']
            text = f"{percent_void}%"
            center_x = (segments[0]['x1'] + segments[-1]['x2']) // 2
            text_position = (int(center_x) - 16, y_min - Y_OFFSET_ABOVE_LINE)

            image = drawer.draw_text(
                image,
                text,
                text_position,
                font_size=FONT_SIZE,
                color=get_void_percent_color(percent_void)
            )

        return image
