# Здесь хранится соответствие: ключ модели -> список названий планограмм (строки для сравнения)

MODEL_PLANOGRAMS = {
    "10104449": [
        "10104449",
    ],
    # "10104443": [],
    # "10104450": [],
}


def get_model_key_by_planogram(planogram_name: str):
    # Ищем ключ модели, у которой название планограммы есть в списке строк для сравнения
    for model_key, planogram_names in MODEL_PLANOGRAMS.items():
        if planogram_name in planogram_names:
            return model_key

    return None
