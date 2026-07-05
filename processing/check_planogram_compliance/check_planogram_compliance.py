
from processing.check_planogram_compliance.markup_processing.shelf_processing import ShelfProcessing
from processing.check_planogram_compliance.drawing_tools.draw_product_information import draw_product_information
from processing.check_planogram_compliance.planogram_comparator import planogram
from processing.check_planogram_compliance.identify_products_for_planogram import identify_products_for_planogram

shelf_processor = ShelfProcessing()

# Идентификация, отрисовка и сравнение с планограммой — выполняется для каждой планограммы изображения
def check_planogram_compliance(image, markup, markup_path, planogram_name):

    # Определение товаров
    all_products_identified = identify_products_for_planogram(image, markup, planogram_name)

    if all_products_identified is None:
        return image, [], []

    # Вывести всю аналитику о товаре на изображение
    image = draw_product_information(image, all_products_identified)

    # Обработка полок
    image, max_percent_void = shelf_processor.process_shelves(
        image,
        markup,
        all_products_identified
    )

    _matches_report, missing_report, _present_report = planogram.comparison(
        all_products_identified,
        markup,
        markup_path
    )

    mismatch_report = planogram.compare_positions(
        all_products_identified,
        markup,
        markup_path
    )


    # (раскоментить на релизе)
    # Скипаем изображение если не привышает минимальный пропуск на полках
    # if percentage_for_notification >= max_percent_void:
    #     continue

    return image, mismatch_report, missing_report
