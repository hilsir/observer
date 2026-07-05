def is_product_on_line(product, line_x_min, line_x_max, line_y_avg, buffer=15):
    """Проверяет, попадает ли товар в область конкретной линии разметки (полки)"""
    # Координаты границ товара
    px1, py1, px2, py2 = product['x1'], product['y1'], product['x2'], product['y2']

    # Проверка: пересекает ли высота полки "тело" товара по вертикали
    # Добавляем небольшой y_buffer (15 пикселей), чтобы компенсировать погрешность разметки
    is_y_hit = (py1 - buffer) <= line_y_avg <= (py2 + buffer)

    # Проверка: находится ли центр товара в горизонтальных пределах линии
    # Тот же буфер, что и по Y, чтобы товар на самом краю линии не вылетал из полки целиком
    product_x_center = (px1 + px2) / 2
    is_x_hit = (line_x_min - buffer) <= product_x_center <= (line_x_max + buffer)

    return is_y_hit and is_x_hit
