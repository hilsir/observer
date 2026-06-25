from identification.goods_identification import Identification
from router_models_config import get_model_key_by_planogram
import os

class ModelManager:
    def __init__(self):
        # Инициализируем модели один раз при создании менеджера
        self._models = {
            # "10104443": Identification(os.getenv("MODEL_10104443_PATH")),
            "10104449": Identification(os.getenv("MODEL_10104449_PATH")),
            # "10104450": Identification(os.getenv("MODEL_10104450_PATH")),
        }

    def get_model_for(self, name: str):
        # Удалить расширение из названия
        name_no_expansion = os.path.splitext(name)[0]
        # Получаем ключ модели через роутер по названию планограммы
        model_key = get_model_key_by_planogram(name_no_expansion)
        if model_key in self._models:
            return self._models[model_key]
