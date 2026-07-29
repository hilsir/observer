# Делит ширину полки (line_x_min..line_x_max) на сегменты слева направо
# пропорционально долям размеров ожидаемых товаров. Сумма размеров товаров
# на полке принимается за 100% её ширины (line_x_max - line_x_min).
# Возвращает список сегментов {'name', 'size', 'x1', 'x2'} — по одному на
# каждый ожидаемый товар, в том же порядке слева направо.
def get_shelf_segments(shelf_expected, line_x_min, line_x_max):
    total_size = sum(product['size'] for product in shelf_expected)
    shelf_width = line_x_max - line_x_min

    segments = []
    cursor = line_x_min

    for product in shelf_expected:
        # Если сумма размеров на полке нулевая (например, полка пуста в плане) — не делим на ноль
        share = (product['size'] / total_size) if total_size > 0 else 0
        segment_width = shelf_width * share

        segments.append({
            'name': product['name'],
            'size': product['size'],
            'x1': cursor,
            'x2': cursor + segment_width,
        })
        cursor += segment_width

    return segments
