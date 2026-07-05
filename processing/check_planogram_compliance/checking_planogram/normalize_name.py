import re


def normalize_name(text):
    """Удаляет спецсимволы, пробелы и приводит к нижнему регистру"""
    if not text:
        return ""
    # Оставляем только буквы и цифры, переводим в нижний регистр
    return re.sub(r'[^a-zA-Zа-яА-Я0-9]', '', str(text)).lower()
