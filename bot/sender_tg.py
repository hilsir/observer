# Библиотека для выполнения HTTP-запросов
import requests
# Библиотека для работы с системными переменными
import os

# Функция для отправки изображения в Telegram
def send_image_to_telegram(file_path):
    # Получение токена бота из переменных окруженияф
    token = os.getenv('BOT_TOKEN')
    # Получение ID группы из переменных окружения
    chat_id = os.getenv('GROUP_ID')
    # Формирование URL адреса для метода отправки фото
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    # Открытие файла изображения в бинарном режиме для чтения
    with open(file_path, 'rb') as photo_file:
        # Отправка POST запроса с файлом и ID чата
        requests.post(url, data={'chat_id': chat_id}, files={'photo': photo_file})


def send_message_arr_to_telegram(missing_array):
    token = os.getenv('BOT_TOKEN')
    chat_id = os.getenv('GROUP_ID')
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    # Формируем заголовок
    text = "❌ *Отсутствует на стеллаже:*\n"

    # Обработка массива
    for i, shelf in enumerate(missing_array):
        if shelf:
            # ОЧЕНЬ ВАЖНО: Экранируем или удаляем символы _, *, [ которые ломают Markdown
            # Просто заменяем их на пробелы для надежности
            clean_products = [str(p).replace("_", " ").replace("*", "").replace("[", "") for p in shelf]
            products_str = ", ".join(clean_products)
            text += f"*Полка {i + 1}:* {products_str}\n"
        else:
            text += f"*Полка {i + 1}:* ✅\n"
    # Обрезаем если большое... Да так много отссутствуе товара
    if len(text) > 3500:
        text = text[:3500] + "...\n\n[Сообщение слишком большое]"

    # Отправка с проверкой результата
    payload = {
        'chat_id': chat_id,
        'text': text,
        'parse_mode': 'Markdown'
    }

    response = requests.post(url, data=payload)


