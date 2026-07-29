
from processing.check_planogram_compliance.drawing_tools.draw_markup_lines import DrawMarkupLines
from processing.check_planogram_compliance.drawing_tools.draw_shelf_segments import DrawShelfSegments
from processing.check_planogram_compliance.drawing_tools.draw_product_information import draw_product_information
from processing.check_planogram_compliance.drawing_tools.draw_void_percent import DrawVoidPercent
from processing.check_planogram_compliance.planogram_comparator import planogram

markup_lines_drawer = DrawMarkupLines()
shelf_segments_drawer = DrawShelfSegments()
void_percent_drawer = DrawVoidPercent()

# Сравнение с планограммой и отрисовка
def check_planogram_compliance(image, all_products_identified, markup, markup_path):

    # Сравнивает распознанные товары с планограммой полка за полкой.
    # Возврощает отчёты
    _matches_report, missing_report, _present_report, mismatch_report, shelf_results = planogram.comparison(
        all_products_identified,
        markup,
        markup_path
    )

    # 1. Тонкая исходная разметка полок (нижний слой)
    image = markup_lines_drawer.draw(image, markup)

    # 2. Красные точки — ориентиры ожидаемых позиций товаров поверх разметки
    image = shelf_segments_drawer.draw(image, shelf_results, markup)

    # 3. Цветные рамки найденных товаров поверх точек-ориентиров
    image = draw_product_information(image, all_products_identified)

    # 4. Процент пустоты полки — маленькой цифрой поверх всего остального
    image = void_percent_drawer.draw(image, shelf_results, markup)

    return image, mismatch_report, missing_report
