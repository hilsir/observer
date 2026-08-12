import json
import os


def _load_model_planograms():
    with open(os.getenv("DATA_ROUTER_PATH"), encoding="utf-8") as f:
        return json.load(f)["models"]


def get_model_key_for_planogram(planogram_name: str):
    # Ищем ключ модели, у которой имя планограммы есть в списке.
    # Читаем data_router.json заново на каждый вызов, чтобы всегда видеть
    # актуальную версию файла, даже если он изменился между итерациями.
    model_planograms = _load_model_planograms()
    for model_key, planogram_names in model_planograms.items():
        if planogram_name in planogram_names:
            return model_key

    return None
