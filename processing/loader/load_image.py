import os
import cv2

images_dir = os.getenv('IMAGES_DIR')


def load_image(img_file_name):
    # путь к картинке (папка + имя файла)
    image_path = os.path.join(images_dir, img_file_name)
    # Загружаем изображение в формате BGR (Для OpenCV)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Пропуск нечитаемого изображения: {img_file_name}")
        return None

    return image
