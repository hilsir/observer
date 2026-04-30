import cv2
import os
import json
from detector import ModelProductDetector

from model_selection import ModelManager
from identification.goods_identification import Identification
from string_formation.create_path import CreatePath
from drawing_tools.draw_product_information import draw_product_information
from processing.markup_processing.shelf_processing import ShelfProcessing
from checking_planogram.compliance_palnogram import CompliancePlanogram
from processing.product_identifier import identify
from dotenv import load_dotenv

load_dotenv()

def image_processing(image_filenames):
    # Берем настройки напрямую из env
    model_goods_path = os.getenv('MODEL_GOODS_PATH')
    detector = ModelProductDetector(model_goods_path)
    shelf_processor = ShelfProcessing()
    planogram = CompliancePlanogram()
    model_manager = ModelManager()
    finished_images = []

    print(f"Найдено изображений для обработки: {len(image_filenames)}")
    # map
    # не перезваписывать img в одну переменую
    for filename in image_filenames:

        # путь к картинке
        image_path = CreatePath.create_path_image(filename)
        # Загружаем изображение в формате BGR (Для OpenCV)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Пропуск нечитаемого изображения: {filename}")
            continue

        # путь к разметке
        markup_path = CreatePath.create_path_markup(filename)

        if not os.path.exists(markup_path):
            print(f"Пропуск нечитаемой разметки: {filename} нет JSON")
            continue

        # Лист массивов координат выделеных областей
        with open(markup_path, 'r') as f:
            markup = json.load(f)

        # Возвращает список словарей с координатами найденных товаров
        all_products = detector.detect(image)
        # Выберает модель для картинки
        selected_model = model_manager.get_model_for(filename)
        # Идентификация найденых товаров через выбраную модель
        all_products_identified = identify(image, all_products, selected_model)
        # Вывессти всю аналитику о таваре на изображение
        image = draw_product_information(image, all_products_identified)

        # обработка полок
        image, max_percent_void = shelf_processor.process_shelves(
            image,
            markup,
            all_products_identified
        )

        matches_report, missing_report, present_report = planogram.comparison(
            all_products_identified,
            markup,
            markup_path
        )

        ## Скипаем изображение если не привышает минимальный пропуск на полках
        ## if percentage_for_notification >= max_percent_void:
        ##     continue

        # Сохраняем результат (имя файла и обработанный кадр) в итоговый список
        finished_images.append((filename, image, missing_report))

    return finished_images
