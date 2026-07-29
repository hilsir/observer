# Товары i-й полки. Если линий разметки больше, чем полок в плане — последняя
# полка плана забирает все лишние линии разом.
def get_shelf_actual_products(array_products_identified, i, planogram_len):
    if i < planogram_len - 1:
        return array_products_identified[i] if i < len(array_products_identified) else []

    shelf_actual = []
    for extra_shelf in array_products_identified[i:]:
        shelf_actual.extend(extra_shelf)
    return shelf_actual
