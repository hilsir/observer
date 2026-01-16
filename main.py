import cv2
from ultralytics import YOLO

# 1. Загружаем модель YOLO (yolov8n.pt - самая быстрая для видео в реальном времени)
model = YOLO('yolov8n.pt')

# 2. Подключаемся к первой доступной камере (0 - стандартная веб-камера)
cap = cv2.VideoCapture(0)

# Проверяем, открылась ли камера
if not cap.isOpened():
    print("Ошибка: Не удалось открыть камеру")
    exit()

while True:
    # 3. Считываем текущий кадр с камеры
    success, frame = cap.read()

    if not success:
        break

    # 4. Прогоняем кадр через нейросеть (stream=True оптимизирует использование памяти)
    results = model(frame, stream=True)

    # Переменная для хранения общей площади обнаруженных товаров
    total_goods_area = 0
    # Площадь всего изображения (кадра)
    frame_area = frame.shape[0] * frame.shape[1]

    # 5. Перебираем результаты детекции
    for r in results:
        # Отрисовываем стандартные рамки YOLO на кадре
        frame = r.plot()

        # 6. Считаем площадь каждой рамки (Box)
        for box in r.boxes.xywh:
            # xywh[2] - ширина, xywh[3] - высота
            w, h = box[2], box[3]
            total_goods_area += (w * h)

    # 7. Рассчитываем процент заполненности (отношение площади товаров к площади кадра)
    # Примечание: 0.8 — это коэффициент поправки, так как между товарами всегда есть зазоры
    occupancy = (total_goods_area / frame_area) * 100
    occupancy = min(float(occupancy), 100.0)  # Ограничиваем до 100%

    # 8. Выводим текст с процентом заполненности на экран
    cv2.putText(frame, f"Filling: {occupancy:.1f}%", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 9. Показываем итоговое окно с видеопотоком
    cv2.imshow("Shelf Scanning", frame)

    # 10. Если нажата клавиша 'q', выходим из цикла
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 11. Освобождаем ресурсы камеры и закрываем все окна
cap.release()
cv2.destroyAllWindows()