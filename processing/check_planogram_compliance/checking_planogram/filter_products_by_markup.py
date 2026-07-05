import cv2


# Оставляет товары, чью рамку пересекает хотя бы одна линия разметки данной планограммы.
def filter_products_by_markup(all_products, markup):

    products_on_markup = []

    for line in markup:
        for product in all_products:
            if product in products_on_markup:
                continue

            rect = (product['x1'], product['y1'], product['x2'] - product['x1'], product['y2'] - product['y1'])

            # Проверяем каждый отрезок линии на пересечение (коллизию) с рамкой товара
            for point_a, point_b in zip(line, line[1:]):
                collided, _, _ = cv2.clipLine(rect, tuple(map(int, point_a)), tuple(map(int, point_b)))
                if collided:
                    products_on_markup.append(product)
                    break

    return products_on_markup
