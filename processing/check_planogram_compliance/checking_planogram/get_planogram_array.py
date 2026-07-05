import numpy as np

from processing.check_planogram_compliance.checking_planogram.is_product_on_line import is_product_on_line


def get_planogram_array(all_products_identified, markup):

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
            if is_product_on_line(product, line_x_min, line_x_max, line_y_avg):
                products_on_current_shelf.append(product)

        # Сортируем товары на конкретной полке строго слева направо
        products_on_current_shelf.sort(key=lambda p: p['x1'])

        # Собираем только названия (имена) товаров для этой строки
        shelf_row = [p.get('name', 'Unknown') for p in products_on_current_shelf]
        planogram_array.append(shelf_row)

    return planogram_array
