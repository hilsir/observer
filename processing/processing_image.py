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
from processing.image_loader import load_image, load_markup
from processing.single_image_pipeline import process_single_image
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

        # Загружаем изображение в формате BGR (Для OpenCV)
        image = load_image(filename)

        if image is None:
            continue

        # Лист массивов координат выделеных областей
        markup, markup_path = load_markup(filename)

        if markup is None:
            continue

        image, mismatch_report = process_single_image(
            filename,
            image,
            markup,
            markup_path,
            detector,
            shelf_processor,
            planogram,
            model_manager
        )

        # Сохраняем результат (имя файла и обработанный кадр) в итоговый список
        finished_images.append((filename, image, mismatch_report))

    return finished_images
