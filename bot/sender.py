# Библиотека для выполнения HTTP-запросов
import requests
# Библиотека для работы с системными переменными
import os

# Функция для отправки изображения в Telegram
def send_image_to_telegram(file_path):
    # Получение токена бота из переменных окружения
    token = os.getenv('BOT_TOKEN')
    # Получение ID группы из переменных окружения
    chat_id = os.getenv('GROUP_ID')
    # Формирование URL адреса для метода отправки фото
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    # Открытие файла изображения в бинарном режиме для чтения
    with open(file_path, 'rb') as photo_file:
        # Отправка POST запроса с файлом и ID чата
        requests.post(url, data={'chat_id': chat_id}, files={'photo': photo_file})