import numpy as np, os, re
from reading_planograms.reading_xlsx import PlanogramReader


class CompliancePlanogram:
    def __init__(self):
        self.planogram = PlanogramReader()

    @staticmethod
    def shelf_format(items):
        """Вспомогательный метод для красивого вывода списка товаров"""
        return ", ".join(items) if items else "---"

    @staticmethod
    def _normalize(text):
        """Удаляет спецсимволы, пробелы и приводит к нижнему регистру"""
        if not text:
            return ""
        # Оставляем только буквы и цифры, переводим в нижний регистр
        return re.sub(r'[^a-zA-Zа-яА-Я0-9]', '', str(text)).lower()

    def comparison(self, all_products_identified, markup, markup_path):
        name_markup = os.path.splitext(os.path.basename(markup_path))[0]
        array_planogram = self.planogram.read_table_to_array(name_markup,"Planogramm")
        array_products_identified = self._get_planogram_array(all_products_identified,markup)

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
            temp_actual_norm = [self._normalize(name) for name in shelf_actual]
            # Оригинальные имена для вывода в отчет
            temp_actual_orig = shelf_actual.copy()

            shelf_missing = []
            shelf_matches = []

            for product_name in shelf_expected:
                norm_expected = self._normalize(product_name)

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



    def _get_planogram_array(self, all_products_identified, markup):

        # Каждая строка массива — это товары на одной полке, отсортированные слева направо.
        planogram_array = []

        for line in markup:
            products_on_current_shelf = []

            # Преобразуем линию в массив numpy для расчетов
            line_np = np.array(line)

            # Определяем границы линии по горизонтали (начало и конец полки)
            line_x_min = np.min(line_np[:, 0])
            line_x_max = np.max(line_np[:, 0])

            # Находим среднюю высоту линии (уровень полки)
            line_y_avg = np.mean(line_np[:, 1])

            for product in all_products_identified:
                # Координаты границ товара
                px1, py1, px2, py2 = product['x1'], product['y1'], product['x2'], product['y2']

                # Проверка: пересекает ли высота полки "тело" товара по вертикали
                # Добавляем небольшой y_buffer (15 пикселей), чтобы компенсировать погрешность разметки
                buffer = 15
                is_y_hit = (py1 - buffer) <= line_y_avg <= (py2 + buffer)

                # Проверка: находится ли центр товара в горизонтальных пределах линии
                product_x_center = (px1 + px2) / 2
                is_x_hit = line_x_min <= product_x_center <= line_x_max

                if is_y_hit and is_x_hit:
                    products_on_current_shelf.append(product)

            # Сортируем товары на конкретной полке строго слева направо
            products_on_current_shelf.sort(key=lambda p: p['x1'])

            # Собираем только названия (имена) товаров для этой строки
            shelf_row = [p.get('name', 'Unknown') for p in products_on_current_shelf]
            planogram_array.append(shelf_row)


        return planogram_array

