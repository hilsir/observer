from ultralytics import YOLO
import os
from dotenv import load_dotenv
load_dotenv()

class ModelProductDetector:
    def __init__(self, model_path):
        # Загружаем модель при инициализации
        self.model = YOLO(model_path)
        self.conf = float(os.getenv('CONF_THRESHOLD'))

    def detect(self, image):
    # Возвращает список словарей с координатами найденных товаров
        results = self.model(image, conf=self.conf, verbose=False)
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                boxes.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'cx': (x1 + x2) // 2,
                    'cy': (y1 + y2) // 2,
                    'w': x2 - x1,
                    'h': y2 - y1
                })
        return boxes