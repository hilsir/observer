import os
import asyncio
from dotenv import load_dotenv
from maxapi import Bot
from maxapi.types import InputMedia

# Загружаем переменные окружения
load_dotenv()

MAX_TOKEN = os.getenv('MAX_BOT_TOKEN')
MAX_GROUP_ID = os.getenv('MAX_GROUP_ID')

# ВАЖНО: Слэш '/' в конце обязателен для работы внутренних механизмов aiohttp
MAX_API_URL = "https://api.max.ru/bot/v1/"


async def _send_photo_async(relative_path):
    # Инициализация бота
    bot = Bot(token=MAX_TOKEN)

    # Установка URL с завершающим слэшем
    if hasattr(bot, 'set_api_url'):
        bot.set_api_url(MAX_API_URL)

    # Формируем абсолютный путь
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.normpath(os.path.join(script_dir, relative_path))

    if not os.path.exists(abs_path):
        print(f"[ERROR] Файл не найден: {abs_path}")
        return

    try:
        print(f"[DEBUG] Попытка отправки через InputMedia: {abs_path}")

        # Согласно документации, Вариант 1: прямая отправка через attachments
        await bot.send_message(
            chat_id=MAX_GROUP_ID,
            attachments=[
                InputMedia(path=abs_path)
            ]
        )
        print(f"[MAX SUCCESS] Фото успешно отправлено!")

    except Exception as e:
        print(f"[MAX ERROR] Ошибка при использовании InputMedia: {e}")

        # Если Вариант 1 не прошел, пробуем Вариант 2 (ручная загрузка)
        try:
            print("[DEBUG] Пробуем Вариант 2: предварительная загрузка медиа...")
            media = InputMedia(path=abs_path)
            attachment = await bot.upload_media(media)
            await bot.send_message(
                chat_id=MAX_GROUP_ID,
                attachments=[attachment]
            )
            print("[MAX SUCCESS] Фото отправлено через upload_media!")
        except Exception as e2:
            print(f"[MAX ERROR] Оба варианта не сработали: {e2}")

    finally:
        # Закрытие сессии
        if hasattr(bot, 'close_session'):
            await bot.close_session()
        elif hasattr(bot, 'session') and bot.session:
            await bot.session.close()


def send_image_to_max(file_path):
    if MAX_TOKEN and MAX_GROUP_ID:
        asyncio.run(_send_photo_async(file_path))


def send_message_arr_to_max(missing_array):
    if not MAX_TOKEN or not MAX_GROUP_ID: return

    text = "❌ **Отсутствует на стеллаже:**\n"
    for i, shelf in enumerate(missing_array):
        items = ", ".join(shelf) if shelf else "✅"
        text += f"**Полка {i + 1}:** {items}\n"

    async def _send_text():
        bot = Bot(token=MAX_TOKEN)
        if hasattr(bot, 'set_api_url'): bot.set_api_url(MAX_API_URL)
        try:
            await bot.send_message(chat_id=MAX_GROUP_ID, text=text)
            print("[MAX SUCCESS] Сообщение отправлено!")
        finally:
            if hasattr(bot, 'close_session'): await bot.close_session()

    asyncio.run(_send_text())


if __name__ == "__main__":
    # Тестовый вызов
    send_image_to_max("../data_for_processing/input_img/10104443.jpg")