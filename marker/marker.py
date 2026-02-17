import cv2
import os
import json

# Настройки
SOURCE_FOLDER = 'for_markup'
MARKUP_FOLDER = './markup'

os.makedirs(MARKUP_FOLDER, exist_ok=True)

# Глобальные переменные для рисования
lines = []  # Список всех готовых линий
current_start = None  # Точка первого клика
temp_img = None


def mouse_event(event, x, y, flags, param):
    global current_start, lines, temp_img

    if event == cv2.EVENT_LBUTTONDOWN:
        if current_start is None:
            # Первый клик — запоминаем начало
            current_start = (x, y)
            cv2.circle(temp_img, current_start, 3, (0, 0, 255), -1)
        else:
            # Второй клик — фиксируем линию
            line = [current_start, (x, y)]
            lines.append(line)
            cv2.line(temp_img, line[0], line[1], (0, 255, 0), 2)
            current_start = None  # Сбрасываем для новой линии

        cv2.imshow("Drawer", temp_img)


def process_images():
    global lines, current_start, temp_img

    files = [f for f in os.listdir(SOURCE_FOLDER) if f.endswith(('.jpg', '.png'))]

    for filename in files:
        # Извлекаем IP (все, что до первого нижнего подчеркивания)
        # Пример: '10.10.44.40_20260120_170050.jpg' -> '10.10.44.40'
        camera_id = filename.split('_')[0]

        img_path = os.path.join(SOURCE_FOLDER, filename)
        # ... далее по тексту без изменений

        img_path = os.path.join(SOURCE_FOLDER, filename)
        original_img = cv2.imread(img_path)
        if original_img is None: continue

        temp_img = original_img.copy()
        lines = []
        current_start = None

        cv2.namedWindow("Drawer")
        cv2.setMouseCallback("Drawer", mouse_event)

        print(f"--- Разметка ID: {camera_id} ---")
        print("1-й клик: начало, 2-й клик: конец линии. S - сохранить, C - сброс, ESC - выход")

        while True:
            cv2.imshow("Drawer", temp_img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):  # Сброс
                temp_img = original_img.copy()
                lines = []
                current_start = None
                print("Очищено")
            elif key == ord('s'):  # Сохранить
                markup_path = os.path.join(MARKUP_FOLDER, f"{camera_id}.json")
                with open(markup_path, 'w') as f:
                    json.dump(lines, f)
                print(f"Сохранено {len(lines)} линий в {camera_id}.json")
                break
            elif key == 27:  # ESC
                return

    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_images()