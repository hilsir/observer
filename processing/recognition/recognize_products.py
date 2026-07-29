import os
from processing.recognition.model_processing.detector import ModelProductDetector
from processing.recognition.model_processing.product_identifier import identify

detector = ModelProductDetector(os.getenv('MODEL_GOODS_PATH'))

def get_recognized_goods(image, selected_model):
    all_detected_products = detector.detect(image)
    return identify(image, all_detected_products, selected_model)

