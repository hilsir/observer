import os

from processing.loader.load_image import load_image
from processing.check_planogram_compliance.compare_image_with_palnograms import compare_image_with_palnograms
from processing.recognition.recognize_products import get_recognized_goods
from router.router_planograms_config import get_planograms_for_image
from processing.get_models_by_planograms import get_models_by_planograms
def images_processing(img_file_names):

    finished_images = []

    print(f"Найдено изображений для обработки: {len(img_file_names)}")

    for group, img_file_name in img_file_names:

        img_file_name_no_ext = os.path.splitext(img_file_name)[0]

        # получаем планограмы прикрепленые к сизображению
        planogram_names = get_planograms_for_image(img_file_name_no_ext)

        if not planogram_names:
            print(f"Пропуск: нет планограмм для {img_file_name}")
            continue

        image = load_image(group, img_file_name)

        if image is None:
            continue

        # Получаем все модели по связаные с планограммами
        selected_model = get_models_by_planograms(planogram_names)

        if selected_model is None:
            print(f"Пропуск фото: не найдена модель идентификации ни для одной из планограмм {planogram_names}")
            continue

        # return (x1,y1,x2,y2,name,confidence)
        all_products_identified = get_recognized_goods(image, selected_model)

        if all_products_identified is None:
            continue

        # Cравнение планограмами
        image, combined_report = compare_image_with_palnograms(
            image,
            all_products_identified,
            planogram_names
        )

        finished_images.append((group, img_file_name, image, combined_report))

    return finished_images
