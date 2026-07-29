import os
import cv2


def save_locally(group, file_name, image, combined_report):

    base_dir = os.environ.get("RETURN_FILES")

    # Собраное имя папки
    folder_name = os.path.splitext(file_name)[0]

    # Полный путь — та же группирующая подпапка, что и у исходного фото
    full_path_folder = os.path.join(base_dir, group, folder_name)

    # Создаем папку, если её нет
    if not os.path.exists(full_path_folder):
        os.makedirs(full_path_folder)

    # Для ромы из будущего, тут проёб с типом дангных. Костыльная склейка листа
    text_str = ""
    for planogram_name, mismatch_report, missing_report in combined_report:
        text_str += f"[{planogram_name}]\n"
        for i, shelf_mismatches in enumerate(mismatch_report):
            if not shelf_mismatches:
                text_str += f"Полка {i + 1}: все позиции совпали\n"
            else:
                mismatches = "; ".join(
                    f"поз. {item['position']}: ожидался '{item['expected']}', найден '{item['actual']}'"
                    for item in shelf_mismatches
                )
                text_str += f"Полка {i + 1}: {mismatches}\n"

            shelf_missing = missing_report[i] if i < len(missing_report) else []
            if not shelf_missing:
                text_str += f"  Отсутствуют: нужные товары присутствуют\n"
            else:
                text_str += f"  Отсутствуют: {', '.join(shelf_missing)}\n"

        text_str += "\n"

    # Сохраняем текст
    report_path = os.path.join(full_path_folder, f"{folder_name}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text_str)

    # Сохранение картинки сразу в финальную папку
    final_image_path = os.path.join(full_path_folder, file_name)
    cv2.imwrite(final_image_path, image)
