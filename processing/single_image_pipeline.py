from drawing_tools.draw_product_information import draw_product_information
from processing.product_identifier import identify


def process_single_image(filename, image, markup, markup_path, detector, shelf_processor, planogram, model_manager):
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

    # matches_report, missing_report, present_report = planogram.comparison(
    #     all_products_identified,
    #     markup,
    #     markup_path
    # )

    mismatch_report = planogram.compare_positions(
        all_products_identified,
        markup,
        markup_path
    )

    ## Скипаем изображение если не привышает минимальный пропуск на полках
    ## if percentage_for_notification >= max_percent_void:
    ##     continue

    return image, mismatch_report
