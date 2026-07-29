# Доля ширины полки, занятая пустыми ('missing') сегментами — показатель
# "пустоты" позиции (вместо старого расчёта по маске пикселей).
def calc_percent_void(segments, line_x_min, line_x_max):
    shelf_width = line_x_max - line_x_min
    if shelf_width <= 0:
        return 0

    missing_width = sum(seg['x2'] - seg['x1'] for seg in segments if seg['status'] == 'missing')
    return round((missing_width / shelf_width) * 100, 1)
