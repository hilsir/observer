import cv2  # Библиотека OpenCV для работы с изображениями и рисования
import numpy as np  # Библиотека для математических вычислений с массивами
from ultralytics import YOLO  # Библиотека для запуска нейросети YOLO
import os  # Модуль для работы с операционной системой (файлами и папками)
from pathlib import Path  # Удобный инструмент для работы с путями к файлам
import json
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN
from scipy.signal import find_peaks

# --- САМЫЕ ПРОСТЫЕ НАСТРОЙКИ ---
MODEL_PATH = '../models/goods.pt'  # Путь к обученной модели нейросети
CONF_THRESHOLD = 0.01 # Порог уверенности: игнорируем объекты, в которых нейросеть уверена меньше чем на 35%
# Чем больше это число, тем легче товары объединяются в одну полку (даже если они криво стоят)
ROW_SENSITIVITY = 1.27
# Зазор под самым нижним товаром в пикселях, чтобы линия полки не «резала» товар
LINE_GAP = 10

INPUT_DIR = '../images'  # Папка с исходными фотографиями
OUTPUT_DIR = '../img_return'  # Папка, куда сохранятся обработанные фото

# Минимальная ширина, чтобы не считать микро-щели (в пикселях)
MIN_VOID_WIDTH = 10
# Допуск по высоте (насколько сосед может быть смещен вертикально)
Y_GAP_ALLOWED = 8

# Порог разницы высот (1.4 = 40%)
HEIGHT_DIFF_THRESHOLD = 1.1

# Сюда будут записываться все найденные аномалии высоты
height_anomalies = []


def is_box_colliding(gap_rect, all_boxes, exclude_ids):
    """Проверяет, не накладывается ли область пустоты на какой-либо существующий товар"""
    gx1, gy1, gx2, gy2 = gap_rect

    for i, b in enumerate(all_boxes):
        if i in exclude_ids:
            continue

        # Находим область пересечения
        ix1 = max(gx1, b['x1'])
        iy1 = max(gy1, b['y1'])
        ix2 = min(gx2, b['x2'])
        iy2 = min(gy2, b['y2'])

        if ix1 < ix2 and iy1 < iy2:
            # Вычисляем площадь пересечения
            intersection_area = (ix2 - ix1) * (iy2 - iy1)
            gap_area = (gx2 - gx1) * (gy2 - gy1)

            # Если товар перекрывает более 10% площади пустоты — удаляем её
            if gap_area > 0 and (intersection_area / gap_area) > 0.1:
                return True
    return False

def is_line_blocking(x_start, x_end, y_center, lines):
    """Проверяет, есть ли нарисованная линия в указанном диапазоне координат"""
    for line in lines:
        # line = [[x1, y1], [x2, y2]]
        lx_min = min(line[0][0], line[1][0])
        lx_max = max(line[0][0], line[1][0])
        ly_min = min(line[0][1], line[1][1])
        ly_max = max(line[0][1], line[1][1])

        # Проверяем пересечение по горизонтали
        if not (lx_max < x_start or lx_min > x_end):
            # Проверяем, находится ли линия на уровне этого товара по высоте
            if ly_min <= y_center <= ly_max or abs((ly_min + ly_max) / 2 - y_center) < Y_GAP_ALLOWED:
                return True
    return False


def process_voids(image, boxes, filename, markup_dir='./markup'):
    global height_anomalies
    camera_id = filename.split('_')[-1].split('.')[0]
    markup_path = os.path.join(markup_dir, f"{camera_id}.json")

    lines = []
    if os.path.exists(markup_path):
        with open(markup_path, 'r') as f:
            lines = json.load(f)

    if not boxes:
        return image

    sorted_boxes = sorted(boxes, key=lambda b: b['x1'])
    processed_intervals = []

    for i, b1 in enumerate(sorted_boxes):
        x_start = int(b1['x2'])
        y_center1 = (b1['y1'] + b1['y2']) / 2
        h1 = b1['y2'] - b1['y1']

        neighbor_idx = -1
        neighbor = None
        for j in range(i + 1, len(sorted_boxes)):
            b2 = sorted_boxes[j]
            y_center2 = (b2['y1'] + b2['y2']) / 2
            if b2['x1'] >= b1['x2'] and abs(y_center1 - y_center2) < Y_GAP_ALLOWED:
                neighbor = b2
                neighbor_idx = j
                break

        if neighbor:
            x_end = int(neighbor['x1'])
            h2 = neighbor['y2'] - neighbor['y1']

            # --- ПРОВЕРКА РАЗНИЦЫ ВЫСОТ ---
            h_max = max(h1, h2)
            h_min = min(h1, h2)
            is_height_issue = False

            if h_min > 0:
                diff_ratio = h_max / h_min
                if diff_ratio >= HEIGHT_DIFF_THRESHOLD:
                    is_height_issue = True
                    # Сохраняем информацию об аномалии
                    height_anomalies.append({
                        'file': filename,
                        'ratio': round(diff_ratio, 2),
                        'coords': (x_start, x_end)
                    })

            # Проверка на дубликаты интервалов
            if any(abs(x_start - p[0]) < 10 and abs(x_end - p[1]) < 10 for p in processed_intervals):
                continue

            gap_width = x_end - x_start
            if gap_width > MIN_VOID_WIDTH:
                y1 = int(max(b1['y1'], neighbor['y1']))
                y2 = int(min(b1['y2'], neighbor['y2']))
                gap_rect = (x_start, y1, x_end, y2)

                if is_line_blocking(x_start, x_end, y_center1, lines):
                    continue

                if is_box_colliding(gap_rect, sorted_boxes, exclude_ids=[i, neighbor_idx]):
                    continue

                # Цвет: Красный для обычного пропуска, Оранжевый для пропуска с разницей высот
                color = (0, 165, 255) if is_height_issue else (0, 0, 255)  # BGR
                label = f"H-DIFF {int(diff_ratio * 100 - 100)}%" if is_height_issue else "GAP"

                overlay = image.copy()
                cv2.rectangle(overlay, (x_start, y1), (x_end, y2), color, -1)
                cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)
                cv2.rectangle(image, (x_start, y1), (x_end, y2), color, 1)
                cv2.putText(image, label, (x_start + 5, y1 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                processed_intervals.append((x_start, x_end))

    return image

def draw_markup_lines(image, filename, markup_dir='./for_markup'):
    """
    Загружает JSON и рисует все сохраненные линии.
    """
    try:
        # Извлекаем номер из имени
        camera_id = filename.split('_')[-1].split('.')[0]
    except:
        return image

    markup_path = os.path.join(markup_dir, f"{camera_id}.json")

    if os.path.exists(markup_path):
        with open(markup_path, 'r') as f:
            saved_lines = json.load(f)

        # Рисуем каждую линию из файла
        for line in saved_lines:
            # line это [(x1, y1), (x2, y2)]
            pt1 = tuple(line[0])
            pt2 = tuple(line[1])
            cv2.line(image, pt1, pt2, (238, 130, 238), 5)  # Желтые линии

        # Пометка ID в углу для контроля
        cv2.putText(image, f"Markup ID: {camera_id}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return image

def run_simple_observer():
    model = YOLO(MODEL_PATH)  # Загружаем нейросеть в память
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)  # Создаем папку для результата, если её еще нет

    extensions = ('.jpg', '.jpeg', '.png')  # Список допустимых расширений файлов
    # Собираем список всех картинок из входной папки
    images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(extensions)]

    for img_name in images:  # Начинаем цикл по каждой картинке
        img_path = os.path.join(INPUT_DIR, img_name)  # Формируем полный путь к файлу
        image = cv2.imread(img_path)  # Читаем (открываем) изображение
        if image is None: continue  # Если файл поврежден или не открылся, пропускаем его

        # ищем объекты на фото
        results = model(image, stream=True, conf=CONF_THRESHOLD, verbose=False)
        boxes = []  # Список, где будем хранить удобные нам данные о рамках товаров
        for r in results:
            for box in r.boxes:  # Перебираем все найденные нейросетью объекты
                # Получаем координаты углов (левый верхний x1,y1 и правый нижний x2,y2)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                # Сохраняем координаты, центр (cx, cy) и высоту (h) товара в словарь
                boxes.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                              'cx': (x1 + x2) // 2, 'cy': (y1 + y2) // 2,
                              'h': y2 - y1})

        # Если нейросеть ничего не нашла
        if not boxes:
            cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), image)  # Просто копируем фото в результат
            continue

        # 1. Рисуем ваши ручные линии
        image = draw_markup_lines(image, img_name)

        # 2. Ищем и обрабатываем пустоты (с учетом коллизий с линиями)
        image = process_voids(image, boxes, img_name)

        # 3. Рисуем рамки товаров
        for b in boxes:
            cv2.rectangle(image, (b['x1'], b['y1']), (b['x2'], b['y2']), (0, 255, 0), 1)

        # 4. Сохраняем
        cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), image)


if __name__ == '__main__':
    run_simple_observer()  # Точка входа: запускаем главную функцию