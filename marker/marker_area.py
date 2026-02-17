import cv2
import os
import json
import numpy as np

# Настройки
SOURCE_FOLDER = 'marker'
MARKUP_FOLDER = 'markup_area'
os.makedirs(MARKUP_FOLDER, exist_ok=True)

# Глобальные переменные
polygons = []  # Список всех готовых областей [[(x1,y1), (x2,y2)...], [...]]
current_poly = []  # Точки текущего (рисуемого) полигона
temp_img = None
original_img = None

def mouse_event(event, x, y, flags, param):
    global current_poly, polygons, temp_img, original_img

    if event == cv2.EVENT_LBUTTONDOWN:
        # Добавляем точку в текущий полигон
        current_poly.append((x, y))

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Замыкаем полигон правым кликом
        if len(current_poly) > 2:
            polygons.append(list(current_poly))
            current_poly = []
            print(f"Область №{len(polygons)} зафиксирована")
        else:
            print("Нужно минимум 3 точки, чтобы создать область!")

    # Обновляем визуализацию
    redraw()


def redraw():
    global temp_img, current_poly, polygons, original_img
    temp_img = original_img.copy()

    # Рисуем уже готовые полигоны (зеленым)
    for poly in polygons:
        pts = np.array(poly, np.int32).reshape((-1, 1, 2))
        cv2.polylines(temp_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        # Заливка (опционально, для наглядности)
        overlay = temp_img.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        cv2.addWeighted(overlay, 0.3, temp_img, 0.7, 0, temp_img)

    # Рисуем линии текущего полигона (красным)
    if len(current_poly) > 0:
        for i in range(len(current_poly) - 1):
            cv2.line(temp_img, current_poly[i], current_poly[i + 1], (0, 0, 255), 2)
        # Рисуем точку у курсора
        cv2.circle(temp_img, current_poly[-1], 4, (0, 0, 255), -1)

    cv2.imshow("Drawer", temp_img)


def process_images():
    global polygons, current_poly, temp_img, original_img

    files = [f for f in os.listdir(SOURCE_FOLDER) if f.endswith(('.jpg', '.png'))]

    for filename in files:
        camera_id = filename.split('_')[-1].split('.')[0]
        img_path = os.path.join(SOURCE_FOLDER, filename)
        original_img = cv2.imread(img_path)
        if original_img is None: continue

        polygons = []
        current_poly = []

        cv2.namedWindow("Drawer")
        cv2.setMouseCallback("Drawer", mouse_event)

        print(f"\n--- Разметка ID: {camera_id} ---")
        print("ЛКМ: поставить точку")
        print("ПКМ: замкнуть область (полигон)")
        print("S: сохранить и перейти к следующему")
        print("C: сброс текущей разметки")
        print("ESC: выход")

        while True:
            redraw()
            key = cv2.waitKey(20) & 0xFF

            if key == ord('c'):  # Сброс
                polygons = []
                current_poly = []
                print("Очищено")
            elif key == ord('s'):  # Сохранить
                markup_path = os.path.join(MARKUP_FOLDER, f"{camera_id}.json")
                with open(markup_path, 'w') as f:
                    json.dump(polygons, f)
                print(f"Сохранено {len(polygons)} областей в {camera_id}.json")
                break
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_images()