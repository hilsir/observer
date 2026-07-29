# Разбирает сырые строки таблицы планограммы (array_planogram) на полки.
# Теперь каждая полка занимает две строки xlsx подряд: сначала имена товаров,
# затем — их размеры в тех же столбцах, например:
#     Товар А | Товар Б | Товар В
#     2       | 2,4     | 6,3
# Возвращает список полок; каждая полка — список {'name': str, 'size': float}
# в том же порядке столбцов (слева направо), что и в исходной таблице.
def get_shelves(raw_rows):
    shelves = []

    for name_row, size_row in zip(raw_rows[0::2], raw_rows[1::2]):
        shelf = [
            {'name': name, 'size': parse_size(size)}
            for name, size in zip(name_row, size_row)
        ]
        shelves.append(shelf)

    return shelves

def parse_size(text):
    """Парсит размер товара из ячейки xlsx: дробная часть может быть записана через запятую."""
    return float(str(text).replace(',', '.'))
