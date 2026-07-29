import os

# images_dir теперь содержит группирующие подпапки (категории), а фото — внутри них.
# Возвращает список (group, filename), оставляя в каждой группе только самый последний
# снимок для каждого уникального IP (camera_id).
def get_img_names(images_dir):

    if not os.path.exists(images_dir):
        print(f"Ошибка: Директория {images_dir} не существует.")
        return []

    groups = [
        d for d in os.listdir(images_dir)
        if os.path.isdir(os.path.join(images_dir, d))
    ]

    result = []

    for group in groups:
        group_dir = os.path.join(images_dir, group)

        # Получаем список всех файлов (фильтруем расширения и исключаем папки)
        raw_images = [
            f for f in os.listdir(group_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
               and os.path.isfile(os.path.join(group_dir, f))
        ]

        # Группируем по IP и выбираем файл с максимальным временем (в названии)
        # Формат: IP_DATE_TIME.jpg
        latest_files_map = {}

        for f in raw_images:
            # Извлекаем IP (часть до первого '_')
            camera_id = f.split('_')[0]

            if camera_id not in latest_files_map:
                latest_files_map[camera_id] = f
            else:
                # Сравниваем строки: т.к. дата ГГГГММДД_ЧЧММСС,
                # более поздняя дата будет лексикографически "больше"
                if f > latest_files_map[camera_id]:
                    latest_files_map[camera_id] = f

        result.extend((group, f) for f in latest_files_map.values())

    return result