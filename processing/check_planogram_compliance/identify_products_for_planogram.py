import os
from processing.check_planogram_compliance.identification.model_selection import ModelManager
from processing.model_processing.product_identifier import identify
from processing.check_planogram_compliance.checking_planogram.filter_products_by_markup import filter_products_by_markup
from processing.model_processing.detector import ModelProductDetector

model_manager = ModelManager()
detector = ModelProductDetector(os.getenv('MODEL_GOODS_PATH'))

# Идентификация товаровпланограммы с её моделью
# Оставляем только те товары, что попадают в разметку
def identify_products_for_planogram(image, markup, planogram_name):
    all_detect_products = detector.detect(image)
    products_on_markup = filter_products_by_markup(all_detect_products, markup)

    # Выбирает модель для планограммы
    selected_model = model_manager.get_model_for(planogram_name)

    if selected_model is None:
        print(f"Пропуск планограммы {planogram_name}: нет модели идентификации")
        return None

    # Идентификация найденых товаров через выбраную модель
    return identify(image, products_on_markup, selected_model)
