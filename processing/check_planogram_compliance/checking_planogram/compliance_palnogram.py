import os
from reading_planograms.reading_xlsx import PlanogramReader
from processing.check_planogram_compliance.checking_planogram.normalize_name import normalize_name
from processing.check_planogram_compliance.checking_planogram.get_planogram_array import get_planogram_array

class CompliancePlanogram:
    def __init__(self):
        self.planogram = PlanogramReader()

    def comparison(self, all_products_identified, markup, markup_path):
        name_markup = os.path.splitext(os.path.basename(markup_path))[0]
        array_planogram = self.planogram.read_table_to_array(name_markup,"Planogramm")
        array_products_identified = get_planogram_array(all_products_identified, markup)

        missing_report = []
        present_report = []
        matches_report = []

        planogram_len = len(array_planogram)

        for i in range(planogram_len):
            shelf_expected = array_planogram[i]

            if i < planogram_len - 1:
                shelf_actual = array_products_identified[i] if i < len(array_products_identified) else []
            else:
                shelf_actual = []
                for extra_shelf in array_products_identified[i:]:
                    shelf_actual.extend(extra_shelf)

            # Создаем список нормализованных названий для текущей полки (факт)
            # Храним кортеж (нормализованное_имя, оригинальное_имя)
            temp_actual_norm = [normalize_name(name) for name in shelf_actual]
            # Оригинальные имена для вывода в отчет
            temp_actual_orig = shelf_actual.copy()

            shelf_missing = []
            shelf_matches = []

            for product_name in shelf_expected:
                norm_expected = normalize_name(product_name)

                if norm_expected in temp_actual_norm:
                    # Находим индекс первого совпадения
                    idx = temp_actual_norm.index(norm_expected)

                    # Добавляем в совпадения (оригинальное имя из плана)
                    shelf_matches.append(product_name)

                    # Удаляем из временных списков, чтобы не посчитать дважды
                    temp_actual_norm.pop(idx)
                    temp_actual_orig.pop(idx)
                else:
                    # Товар не найден даже после нормализации
                    shelf_missing.append(product_name)

            missing_report.append(shelf_missing)
            matches_report.append(shelf_matches)
            present_report.append(temp_actual_orig)  # Здесь остаются оригинальные лишние товары

        return matches_report, missing_report, present_report



    def compare_positions(self, all_products_identified, markup, markup_path):
        """
        (Нагенерено)
        Дословно сравнивает порядок товаров на полке (слева-направо) с планограммой.
        В отличие от comparison(), здесь не учитываются отсутствующие или лишние товары —
        отмечаются только случаи, когда товар есть на полке, но стоит не на своей позиции.
        """
        name_markup = os.path.splitext(os.path.basename(markup_path))[0]
        array_planogram = self.planogram.read_table_to_array(name_markup, "Planogramm")
        array_products_identified = get_planogram_array(all_products_identified, markup)

        mismatch_report = []
        planogram_len = len(array_planogram)

        for i in range(planogram_len):
            shelf_expected = array_planogram[i]

            if i < planogram_len - 1:
                shelf_actual = array_products_identified[i] if i < len(array_products_identified) else []
            else:
                shelf_actual = []
                for extra_shelf in array_products_identified[i:]:
                    shelf_actual.extend(extra_shelf)

            norm_expected = [normalize_name(name) for name in shelf_expected]
            norm_actual = [normalize_name(name) for name in shelf_actual]

            shelf_mismatches = []

            for pos in range(min(len(norm_expected), len(norm_actual))):
                if norm_expected[pos] == norm_actual[pos]:
                    continue

                shelf_mismatches.append({
                    'position': pos + 1,
                    'expected': shelf_expected[pos],
                    'actual': shelf_actual[pos],
                })

            mismatch_report.append(shelf_mismatches)

        return mismatch_report
