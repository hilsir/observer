# Здесь хранится соответствие: ключ модели -> список планограмм (строки для сравнения)

MODEL_PLANOGRAMS = {
    "10104449": [
    #     "Каши смеси 3",
    #     "Вода_смеси_1",
    ],
    "10104450": [
        "Кофе растворимый_3",
        "Кофе молотый_растворимый_2",
        "Кофе_зерновой_растворимый_1",
    ],
}


def get_model_key_for_planogram(planogram_name: str):
    # Ищем ключ модели, у которой имя планограммы есть в списке
    for model_key, planogram_names in MODEL_PLANOGRAMS.items():
        if planogram_name in planogram_names:
            return model_key

    return None
