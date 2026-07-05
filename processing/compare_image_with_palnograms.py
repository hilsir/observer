from processing.loader.load_markup_by_name import load_markup_by_name
from processing.check_planogram_compliance.check_planogram_compliance import check_planogram_compliance

# Накладывает на изображение все его планограммы c разметкой
def compare_image_with_palnograms(image, planogram_names):

    combined_report = []

    # Все разметки накладываются на одно изображение
    for planogram_name in planogram_names:

        # Лист массивов координат выделеных областей для этой планограммы
        markup, markup_path = load_markup_by_name(planogram_name)

        if markup is None:
            continue

        # Сравнить изображение с одной планограмой
        image, mismatch_report, missing_report = check_planogram_compliance(
            image,
            markup,
            markup_path,
            planogram_name
        )

        combined_report.append((planogram_name, mismatch_report, missing_report))

    return image, combined_report
