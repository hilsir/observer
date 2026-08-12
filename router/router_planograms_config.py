import json
import os


def get_planograms_for_image(image_name: str):
    # Возвращает список названий планограмм для изображения.
    # Читаем data_router.json заново на каждый вызов, чтобы всегда видеть
    # актуальную версию файла, даже если он изменился между итерациями.
    with open(os.getenv("DATA_ROUTER_PATH"), encoding="utf-8") as f:
        image_planograms = json.load(f)["images"]

    return image_planograms.get(image_name, [])
