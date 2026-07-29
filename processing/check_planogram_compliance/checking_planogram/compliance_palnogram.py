import os
from processing.loader.load_xlsx import PlanogramReader
from processing.check_planogram_compliance.checking_planogram.get_planogram_array.get_planogram_array import get_planogram_array
from processing.check_planogram_compliance.checking_planogram.get_shelves.get_shelves import get_shelves
from processing.check_planogram_compliance.checking_planogram.get_shelf_segments import get_shelf_segments
from processing.check_planogram_compliance.checking_planogram.comparison.get_shelf_actual_products import get_shelf_actual_products
from processing.check_planogram_compliance.checking_planogram.comparison.get_line_bounds import get_line_bounds
from processing.check_planogram_compliance.checking_planogram.comparison.match_shelf_segments import match_shelf_segments
from processing.check_planogram_compliance.checking_planogram.comparison.calc_percent_void import calc_percent_void


class CompliancePlanogram:
    def __init__(self):
        self.planogram = PlanogramReader()  # читает xlsx-планограммы

    # Сравнивает распознанные товары с планограммой полка за полкой.
    def comparison(self, all_products_identified, markup, markup_path):

        # Читаем план (xlsx) и группируем распознанные товары по полкам разметки
        # Имя файла без расширения -> str
        name_markup = os.path.splitext(os.path.basename(markup_path))[0]
        # Строки с таблицы -> list[list[str]]: строка таблицы, ячейка — строка
        rows_with_xlsx = self.planogram.read_table_to_array(name_markup)
        # Полки: имя+размер товара -> list[list[{'name': str, 'size': float}]]
        array_planogram = get_shelves(rows_with_xlsx)
        # Товары по полкам разметки -> list[list[dict]]: словарь товара (x1,y1,x2,y2,name,confidence);
        array_products_identified = get_planogram_array(all_products_identified, markup)

        missing_report = []
        present_report = []
        matches_report = []
        mismatch_report = []
        shelf_results = []

        planogram_len = len(array_planogram)

        for i in range(planogram_len):

            # Линия разметки для этой полки; если линий меньше, чем полок в плане — берём последнюю
            line = markup[i] if i < len(markup) else markup[-1]
            # левый и правый край полки
            line_x_min, line_x_max = get_line_bounds(line)


            shelf_expected = array_planogram[i]
            # Делит полку на сегменты по размерам товаров -> list[{'name': str, 'size': float, 'x1': float, 'x2': float}]
            # x1,x2 позиция сегмента
            segments = get_shelf_segments(shelf_expected, line_x_min, line_x_max)

            # товары на этой полке
            shelf_actual = get_shelf_actual_products(array_products_identified, i, planogram_len)

            # Сравнивает сегменты с товарами по пересечению: каждому сегменту — свой товар или пусто.
            # shelf_missing/matches — имена товаров не найденые/найденые,
            # shelf_mismatches — {position,expected,actual},
            # - position — номер позиции на полке (считая с 1)
            # - expected — что должно было стоять по плану
            # - actual — что реально нашлось в этом месте
            # extra_products — товары, не попавшие ни в один сегмент (лишние)
            shelf_missing, shelf_matches, shelf_mismatches, extra_products = match_shelf_segments(segments, shelf_actual)

            # Товары на полке, не привязанные ни к одному сегменту — лишние, на отрисовке жёлтые
            for extra_product in extra_products:
                extra_product['status'] = 'mismatch'

            missing_report.append(shelf_missing)      # имена товаров, которых не нашлось
            matches_report.append(shelf_matches)      # имена товаров, найденных на своём месте
            mismatch_report.append(shelf_mismatches)  # товар есть, но не тот: {position, expected, actual}

            extra_names = [product.get('name', 'Unknown') for product in extra_products]  # имена лишних товаров
            present_report.append(extra_names)

            shelf_results.append({
                'segments': segments,                                                   # сегменты полки со статусами — для отрисовки
                'percent_void': calc_percent_void(segments, line_x_min, line_x_max),    # процент пустоты полки
            })

        return matches_report, missing_report, present_report, mismatch_report, shelf_results
