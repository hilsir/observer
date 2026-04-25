import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# Тут немного нагенерила нейронка. Не хочу в этом разбиратся
class TextDrawer:
    def __init__(self, font_name="DejaVuSans-Bold.ttf", default_size=14):
        # Путь к проекту
        project_root = Path(__file__).resolve().parent.parent
        self.font_path = project_root / "fonts" / font_name
        # Путь к шрифту: Корень -> fonts -> файл
        self.font_path = os.path.join(project_root, "fonts", font_name)
        self.default_size = default_size
        self._fonts = {}

        if not os.path.exists(self.font_path):
            print(f"Шрифт не найден по пути: {self.font_path}")

    def _get_font(self, size):
        if size not in self._fonts:
            try:
                if os.path.exists(self.font_path):
                    self._fonts[size] = ImageFont.truetype(self.font_path, size)
                else:
                    self._fonts[size] = ImageFont.load_default()
            except Exception:
                self._fonts[size] = ImageFont.load_default()
        return self._fonts[size]

    def draw_text(self, img, text, position, font_size=None, color=(255, 255, 0), with_bg=True):
        if img is None: return None

        size = font_size or self.default_size
        font = self._get_font(size)

        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        if with_bg:
            # textbbox возвращает (left, top, right, bottom)
            bbox = draw.textbbox(position, text, font=font)
            draw.rectangle(bbox, fill=(0, 0, 0))

        draw.text(position, text, font=font, fill=color)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)