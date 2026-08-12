import os
from dotenv import load_dotenv
load_dotenv()

# заставить систему думать, что архитектура gfx1100 (для федоры) на серваке коментить
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import time
from datetime import datetime
from processing.images_processing import images_processing
from image_filter import get_img_names
from save_result.send_files import save_locally
import zoneinfo
from processing.get_models_by_planograms import model_manager

def main():
    print("start")

    images_dir = os.getenv('IMAGES_DIR')

    # Список нужных моментов (Иркутское время)
    target_times = [t.strip() for t in os.getenv("TIME_MESSAGES").split(",")]
    last_run_time = ""  # Чтобы не срабатывало дважды в одну и ту же минуту

    models_reload_time = os.getenv("MODELS_RELOAD_TIME")
    last_reload_time = ""  # Чтобы не срабатывало дважды в одну и ту же минуту

    while True:

        # Устанавливаем смещение для Иркутска
        irk_tz = zoneinfo.ZoneInfo("Asia/Irkutsk")
        now_irk = datetime.now(irk_tz)
        current_time_str = now_irk.strftime("%H:%M")

        # Убиваем кеш со всмеи моделями для подгрузки изменений и новых моделией
        if current_time_str == models_reload_time and current_time_str != last_reload_time:
            model_manager.reset()
            last_reload_time = current_time_str
            print("Кеш моделей идентификации сброшен по расписанию")

        # Проверяем, совпадает ли время и не запускались ли мы уже в эту минуту
        if current_time_str in target_times and current_time_str != last_run_time:
            start(images_dir)
            last_run_time = current_time_str
            print(f"Обработка завершена. Следующая проверка по расписанию...")

        # Проверяем время каждые 30 секунд
        time.sleep(1)

def start(images_dir):
    image_file_names = get_img_names(images_dir)

    if image_file_names:
        # Основная обработка
        finished_images = images_processing(image_file_names)

        # Сохранение
        for group, filename, image, missing_report in finished_images:
            save_locally(group, filename, image, missing_report)

            print(f"Готово: {group}/{filename}")
            time.sleep(5)

def run_once():
    images_dir = os.getenv('IMAGES_DIR')
    start(images_dir)

if __name__ == "__main__":
    main()
    # Запуск без ожидания времени (для тестов)
    # run_once()