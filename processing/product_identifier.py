import io
import cv2
from PIL import Image

"""
Вырезает области товаров на изображении и определяет их названия

Args:
    image: исходное изображение (numpy array, BGR формат)
    all_products: список словарей с координатами товаров (x1, y1, x2, y2)
    identifier: объект для распознавания с методом predict()

Returns:
    list: бновлёный список all_products с полями name и confidence
"""

def identify(image, all_products, identifier):

    for p in all_products:
        # Вырезаем область товара
        crop_bgr = image[p['y1']:p['y2'], p['x1']:p['x2']]

        # Проверка что не битое
        if crop_bgr.size > 0:
            # Конвертируем BGR в RGB (т.к. PIL и нейросети работают в RGB)
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

            # Превращаем массив в объект PIL Image
            pil_img = Image.fromarray(crop_rgb)

            #Создаем "виртуальный файл" в оперативной памяти
            img_buffer = io.BytesIO()

            # Сохраняем картинку в этот буфер (в формате PNG или JPEG)
            pil_img.save(img_buffer, format='PNG')

            # Перематываем "курсор" буфера в начало
            img_buffer.seek(0)

            # Передаем этот буфер в predict
            # Identification внутри сделает Image.open(img_buffer)
            # и это сработает, так как Image.open понимает объекты-файлы

            name, confidence = identifier.predict(img_buffer)
            p['name'] = name
            p['confidence'] = confidence
        else:
            p['name'] = "Empty"
            p['confidence'] = 0.0

    # Сортировка (не обязательна)
    all_products.sort(key=lambda x: x['x1'])

    return all_products