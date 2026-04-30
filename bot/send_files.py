import os


def save_locally(save_path, missing_report):

    base_dir = os.environ.get("RETURN_FILES")

    # Имя картинки без расширения
    file_name = os.path.basename(save_path)
    # Собраное имя папки
    folder_name = os.path.splitext(file_name)[0]

    # Полный путь
    full_path_folder = os.path.join(base_dir, folder_name)

    # Создаем папку, если её нет
    if not os.path.exists(full_path_folder):
        os.makedirs(full_path_folder)

    # Сохраняем текст
    report_path = os.path.join(full_path_folder,f"{folder_name}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        if isinstance(missing_report, list):
            f.write("\n".join(str(item) for item in missing_report))
        else:
            f.write(str(missing_report))

    # Новый к путь картинке
    final_image_path = os.path.join(full_path_folder, file_name)
    # Сохранение картинки
    os.replace(save_path, final_image_path)