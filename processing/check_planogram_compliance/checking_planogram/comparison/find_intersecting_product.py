from processing.check_planogram_compliance.checking_planogram.comparison.normalize_name import normalize_name


# Совпадение с сегментом полки считается по пересечению рамок по ширине, а не по индексу/порядку.
# Среди товаров, чья рамка вообще пересекается с сегментом:
#   - если среди них есть товар с ИМЕНЕМ, совпадающим с ожидаемым для сегмента — берём его
#     (даже если он пересекается меньше, чем случайно залезший сосед) — это даёт 'match';
#   - если такого нет ни одного — берём тот, что пересекается сильнее всего, вне зависимости
#     от имени — это даёт 'mismatch'.
# Возвращает индекс в products, либо None, если пересечений нет вовсе.
def find_intersecting_product(segment, products):
    norm_expected = normalize_name(segment['name'])

    best_idx = None
    best_overlap = 0
    best_match_idx = None
    best_match_overlap = 0

    for idx, product in enumerate(products):
        overlap = min(segment['x2'], product['x2']) - max(segment['x1'], product['x1'])
        if overlap <= 0:
            continue

        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = idx

        if normalize_name(product.get('name', '')) == norm_expected and overlap > best_match_overlap:
            best_match_overlap = overlap
            best_match_idx = idx

    return best_match_idx if best_match_idx is not None else best_idx
