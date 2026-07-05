import os

from processing.loader.load_image import load_image
from processing.compare_image_with_palnograms import compare_image_with_palnograms
from router.router_planograms_config import get_planograms_for_image

def image_processing(img_file_names):

    finished_images = []

    print(f"Найдено изображений для обработки: {len(img_file_names)}")

    for img_file_name in img_file_names:

        image = load_image(img_file_name)

        if image is None:
            continue

        filename_no_ext = os.path.splitext(img_file_name)[0]
        # получаем все планограмы прикрепленые к сизображению
        planogram_names = get_planograms_for_image(filename_no_ext)

        if not planogram_names:
            print(f"Пропуск: нет планограмм для {img_file_name}")
            continue

        # старт сравнения планограмами
        image, combined_report = compare_image_with_palnograms(
            image,
            planogram_names
        )

        finished_images.append((img_file_name, image, combined_report))

    return finished_images
