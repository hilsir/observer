import cv2
import os
import numpy as np
from dotenv import load_dotenv


# Нижний слой: тонкая линия разметки полки, без подсветки пустот
class DrawMarkupLines:
    def __init__(self):
        load_dotenv()
        self.color_line = tuple(map(int, os.getenv('COLOR_LINE').split(',')))

    def draw(self, image, markup):
        for line in markup:
            line_np = np.array(line, np.int32)
            cv2.polylines(image, [line_np], isClosed=False, color=self.color_line, thickness=1)
        return image
