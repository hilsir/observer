import cv2
import os
import time
from dotenv import load_dotenv
from processing import image_processing
from bot.sender import send_image_to_telegram
from image_filter import get_latest_images

load_dotenv()

def main():
    # Папка с изображениями
    images_dir = os.getenv('IMAGES_DIR')
    # Папка с обработанными изображениями
    images_return_dir = os.getenv('IMG_RETURN_DIR')

    # Время между сохранениями
    interval_minutes = int(os.getenv('TIME_BETWEEN_CHECKS'))
    interval_seconds = interval_minutes * 60

    # создать если нет
    os.makedirs(images_return_dir, exist_ok=True)

    while True:
        images = get_latest_images(images_dir)

        if images:
            # Обработка
            finished_images = image_processing(images)

            # Сохранение
            for filename, image in finished_images:
                save_path = os.path.join(images_return_dir, filename)
                cv2.imwrite(save_path, image)
                send_image_to_telegram(save_path)
                print(f"Готово: {filename}")

        print(f"Ожидаю {interval_minutes} мин. до следующей проверки...")
        time.sleep(interval_seconds)

if __name__ == "__main__":
    main()