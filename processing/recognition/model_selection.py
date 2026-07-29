from processing.recognition.goods_identification import Identification
from router.router_models_config import get_model_key_for_planogram, MODEL_PLANOGRAMS
import os

class ModelManager:
    def __init__(self):
        models_dir = os.getenv("MODES_PATH")
        # Заргужаем все модели из списка
        self._models = {
            model_key: Identification(f"{models_dir}/model_{model_key}.pth")
            for model_key in MODEL_PLANOGRAMS
        }

    def get_model_for(self, planogram_name: str):
        # Удалить расширение из названия
        planogram_name_no_exp = os.path.splitext(planogram_name)[0]
        # Получаем ключ модели через роутер по имени планограммы
        model_key = get_model_key_for_planogram(planogram_name_no_exp)

        if model_key is None:
            print(f"Модель не найдена: планограмма '{planogram_name}' не привязана ни к одному ключу модели")
            return None

        if model_key not in self._models:
            print(f"Модель не найдена: ключ '{model_key}' отсутствует среди загруженных моделей")
            return None

        return self._models[model_key]
