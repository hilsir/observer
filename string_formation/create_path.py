import os
import cv2

images_dir = os.getenv('IMAGES_DIR')
markup_dir = os.getenv('MARKUP_DIR')

class CreatePath:
    @staticmethod
    def create_path_image(filename):
        # Собираем полный путь к картинке (папка + имя файла)
        return os.path.join(images_dir, filename)

    @staticmethod
    def create_path_markup(filename):

        # Неактуальное названиее НО могут вернуть не УДАЛЯТЬ !!!

        # # Извлекаем ID камеры из названия
        # camera_id = filename.split('_')[0]
        # # Убираем точки из имени файла (если они есть)
        # camera_id = camera_id.replace('.', '').replace(',', '').replace(';', '')
        # # Путь к JSON-файлу с разметкой
        # return os.path.join(markup_dir, f"{camera_id}.json")

        # Убрать расширениеп из названия
        filename_no_expansion = os.path.splitext(filename)[0]
        # Путь к JSON-файлу с разметкой
        print(f"{filename_no_expansion}.json")
        return os.path.join(markup_dir, f"{filename_no_expansion}.json")
