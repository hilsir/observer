from processing.loader.load_markup import load_markup
from processing.check_planogram_compliance.check_planogram_compliance import check_planogram_compliance

# Накладывает на изображение все планограммы c разметкой
def compare_image_with_palnograms(image, all_products_identified, planogram_names):

    combined_report = []

    # Все разметки накладываются на одно изображение
    for planogram_name in planogram_names:

        # Лист разметок для этой планограммы (небольошое исчключение из архитектуры чтоб не нагружать кодом)
        markup, markup_path = load_markup(planogram_name)

        if markup is None:
            continue

        # Сравнить изображение с одной планограмой
        image, mismatch_report, missing_report = check_planogram_compliance(
            image,
            all_products_identified,
            markup,
            markup_path
        )

        combined_report.append((planogram_name, mismatch_report, missing_report))

    return image, combined_report
