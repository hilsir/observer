from identification.goods_identification import Identification
import os

class ModelManager:
    def __init__(self):
        # Инициализируем модели один раз при создании менеджера
        self._models = {
            "10104443": Identification(os.getenv("MODEL_10104443_PATH")),
            "10104449": Identification(os.getenv("MODEL_10104449_PATH")),
            "10104450": Identification(os.getenv("MODEL_10104450_PATH")),
        }

    def get_model_for(self, name: str):
        # Удалить расширение из названия
        name_no_expansion = os.path.splitext(name)[0]
        if name_no_expansion in self._models:
            return self._models[name_no_expansion]
