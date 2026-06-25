import os
import time
from datetime import datetime, timedelta, timezone
from processing.processing_image import image_processing
from image_filter_old import get_latest_images
from bot.send_files import save_locally
from dotenv import load_dotenv
import zoneinfo

# заставить систему думать, что архитектура gfx1100 (для федоры) на серваке коментить
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

load_dotenv()

def main():
    print("start")
    # Папка с изображениями
    images_dir = os.getenv('IMAGES_DIR')

    # Список нужных моментов (Иркутское время)
    target_times = [t.strip() for t in os.getenv("TIME_MESSAGES").split(",")]
    last_run_time = ""  # Чтобы не срабатывало дважды в одну и ту же минуту

    while True:
        # Устанавливаем смещение для Иркутска
        irk_tz = zoneinfo.ZoneInfo("Asia/Irkutsk")
        now_irk = datetime.now(irk_tz)
        current_time_str = now_irk.strftime("%H:%M")

        # Проверяем, совпадает ли время и не запускались ли мы уже в эту минуту
        if current_time_str in target_times and current_time_str != last_run_time:
            start(images_dir)
            last_run_time = current_time_str
            print(f"Обработка завершена. Следующая проверка по расписанию...")

        # Проверяем время каждые 30 секунд
        time.sleep(1)

def start(images_dir):
    images = get_latest_images(images_dir)

    if images:
        # Обработка
        finished_images = image_processing(images)

        # Сохранение
        for filename, image, missing_report in finished_images:
            save_locally(filename, image, missing_report)

            print(f"Готово: {filename}")
            time.sleep(5)

def run_once():
    # Запуск без ожидания времени (для тестов)
    images_dir = os.getenv('IMAGES_DIR')
    start(images_dir)

if __name__ == "__main__":
    main()
    # run_once()