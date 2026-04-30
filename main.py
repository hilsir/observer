import cv2
import os
import time
from datetime import datetime, timedelta, timezone
from processing.processing_image import image_processing
from bot.sender_tg import send_image_to_telegram,send_message_arr_to_telegram
from image_filter_old import get_latest_images
from bot.send_files import save_locally
from dotenv import load_dotenv
import zoneinfo

# заставить систему думать, что у вас архитектура gfx1100 (для федоры) на серваке коментить
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

load_dotenv()

def main():
    print("start")
    # Папка с изображениями
    images_dir = os.getenv('IMAGES_DIR')
    # Папка с обработанными изображениями
    images_return_dir = os.getenv('IMG_RETURN_DIR')

    # Список нужных моментов (Иркутское время)
    target_times = [t.strip() for t in os.getenv("TIME_MESSAGES").split(",")]
    last_run_time = ""  # Чтобы не срабатывало дважды в одну и ту же минуту

    # создать путь если нет
    os.makedirs(images_return_dir, exist_ok=True)

    while True:
        # Устанавливаем смещение для Иркутска
        irk_tz = zoneinfo.ZoneInfo("Asia/Irkutsk")
        now_irk = datetime.now(irk_tz)
        current_time_str = now_irk.strftime("%H:%M")

        # Проверяем, совпадает ли время и не запускались ли мы уже в эту минуту
        if current_time_str in target_times and current_time_str != last_run_time:
            images = get_latest_images(images_dir)

            if images:
                # Обработка
                finished_images = image_processing(images)

                # Сохранение
                for filename, image, missing_report in finished_images:
                    save_path = os.path.join(images_return_dir, filename)
                    cv2.imwrite(save_path, image)
                    save_locally(save_path, missing_report)

                    # Пока без телеграмма
                    # send_image_to_telegram(save_path)
                    # send_message_arr_to_telegram(missing_report)

                    print(f"Готово: {filename}")
                    time.sleep(5)
            last_run_time = current_time_str
            print(f"Обработка завершена. Следующая проверка по расписанию...")

        # Проверяем время каждые 30 секунд
        time.sleep(1)

if __name__ == "__main__":
    main()