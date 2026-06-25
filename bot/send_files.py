import os
import cv2


def save_locally(file_name, image, missing_report):

    base_dir = os.environ.get("RETURN_FILES")

    # Собраное имя папки
    folder_name = os.path.splitext(file_name)[0]

    # Полный путь
    full_path_folder = os.path.join(base_dir, folder_name)

    # Создаем папку, если её нет
    if not os.path.exists(full_path_folder):
        os.makedirs(full_path_folder)

    # Для ромы из будущего, тут проёб с типом дангных. Костыльная склейка листа
    text_str = ""
    for i, shelf in enumerate(missing_report):
        if not shelf:
            text_str += f"Полка {i + 1}: ---\n"
            continue
        mismatches = "; ".join(
            f"поз. {item['position']}: ожидался '{item['expected']}', найден '{item['actual']}'"
            for item in shelf
        )
        text_str += f"Полка {i + 1}: {mismatches}\n"

    # Обрезаем
    if len(text_str) > 3500:
        text_str = text_str[:3500] + "...\n\n[Сообщение слишком большое]"

    # Сохраняем текст
    report_path = os.path.join(full_path_folder, f"{folder_name}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text_str)

    # Сохранение картинки сразу в финальную папку
    final_image_path = os.path.join(full_path_folder, file_name)
    cv2.imwrite(final_image_path, image)