from processing.check_planogram_compliance.checking_planogram.comparison.find_intersecting_product import find_intersecting_product
from processing.check_planogram_compliance.checking_planogram.comparison.normalize_name import normalize_name


# Разбирает сегменты полки по очереди: для каждого ищет пересекающийся товар
# (find_intersecting_product) и сравнивает его имя с ожидаемым. Статус пишется прямо
# в сегмент и в найденный товар (product['status']) — на этих же объектах дальше
# работает отрисовка.
# Возвращает (shelf_missing, shelf_matches, shelf_mismatches, extra_products) —
# extra_products — товары, оставшиеся непривязанными ни к одному сегменту.
def match_shelf_segments(segments, shelf_actual):
    remaining_actual = list(shelf_actual)

    shelf_missing = []
    shelf_matches = []
    shelf_mismatches = []

    for pos, segment in enumerate(segments):
        found_idx = find_intersecting_product(segment, remaining_actual)

        if found_idx is None:
            segment['status'] = 'missing'
            shelf_missing.append(segment['name'])
            continue

        found_product = remaining_actual.pop(found_idx)
        segment['product'] = found_product

        if normalize_name(segment['name']) == normalize_name(found_product.get('name', '')):
            segment['status'] = 'match'
            found_product['status'] = 'match'
            shelf_matches.append(segment['name'])
        else:
            segment['status'] = 'mismatch'
            found_product['status'] = 'mismatch'
            shelf_mismatches.append({
                'position': pos + 1,
                'expected': segment['name'],
                'actual': found_product.get('name', ''),
            })

    return shelf_missing, shelf_matches, shelf_mismatches, remaining_actual
