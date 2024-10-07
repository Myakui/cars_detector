import cv2
import numpy as np
from ultralytics import YOLO

# Глобальные переменные
regions = {'red': [], 'blue': [], 'green': [], 'black': []}
car_count = {'red': 0, 'blue': 0, 'green': 0, 'black': 0}
drawing = False  # Индикатор рисования
current_color_idx = 0  # Индекс текущего цвета
scale_factor = 1.5  # Масштабирование видео

# Словарь цветов для отображения зон
color_dict = {'red': (0, 0, 255), 'blue': (255, 0, 0), 'green': (0, 255, 0), 'black': (0, 0, 0)}

# Функция для рисования зон
def draw_regions(image):
    for color, pts in regions.items():
        if len(pts) > 1:
            cv2.polylines(image, [np.array(pts)], isClosed=True, color=color_dict[color], thickness=2)

# Функция для обработки нажатия мыши и добавления точек в зону
def select_region(event, x, y, flags, param):
    global drawing, current_color_idx
    current_color = list(regions.keys())[current_color_idx]

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        # Добавляем точку в текущую зону
        if len(regions[current_color]) < 4:  # Ограничение в 4 точки для зоны
            regions[current_color].append((x, y))

        # После того как для текущего цвета установлено 4 точки, переключаемся на следующий цвет
        if len(regions[current_color]) == 4:
            current_color_idx = (current_color_idx + 1) % len(regions)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# Функция загрузки модели YOLO
def load_yolo_model():
    model = YOLO('yolov8n.pt')  # Загрузите модель YOLOv8
    return model

# Функция определения машин
def detect_cars(frame, model):
    results = model(frame)
    detections = []
    # Получение координат обнаруженных объектов в формате (x1, y1, x2, y2)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Выбор координат прямоугольников
        classes = result.boxes.cls.cpu().numpy()  # Классы объектов
        confidences = result.boxes.conf.cpu().numpy()  # Уверенность
        for box, cls, conf in zip(boxes, classes, confidences):
            if int(cls) == 2 and conf > 0.5:  # ID класса 2 - машины
                x1, y1, x2, y2 = map(int, box[:4])
                detections.append((x1, y1, x2 - x1, y2 - y1))
    return detections

# Функция проверки, находится ли точка в зоне
def is_inside_zone(center, zone):
    if len(zone) == 0:  # Проверка на наличие точек в зоне
        return False
    return cv2.pointPolygonTest(np.array(zone, dtype=np.int32), center, False) >= 0

# Основная функция
def main():
    global current_color_idx
    model = load_yolo_model()

    print("Укажите путь до видео")
    path = input()
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видео.")
        return

    # Установка окна для выбора зон и детекции машин (одно окно)
    cv2.namedWindow('Car Detection', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Car Detection', select_region)

    print("Используйте ЛКМ для добавления точек в зону. Нажмите 'q', чтобы завершить.")

    # Цикл для выбора зон
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: Не удалось прочитать кадр.")
            break

        # Масштабируем видео
        frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

        # Рисуем зоны
        draw_regions(frame)

        # Отображаем кадр
        cv2.imshow('Car Detection', frame)

        # Завершение программы только при нажатии на 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Проверка, что зоны были определены
    if not any([len(pts) == 4 for pts in regions.values()]):
        print("Не были заданы зоны для подсчета машин.")
        return

    # Основной цикл для детекции машин, без открытия нового окна
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

        # Рисуем зоны перед детекцией
        draw_regions(frame)

        detections = detect_cars(frame, model)

        for (x, y, w, h) in detections:
            center = (x + w // 2, y + h // 2)
            for color in regions:
                if is_inside_zone(center, regions[color]):
                    car_count[color] += 1
                    break

            # Рисуем прямоугольники вокруг обнаруженных машин
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Отображение счетчиков
        for color, count in car_count.items():
            cv2.putText(frame, f'{color.capitalize()}: {count}', (10, 30 + list(car_count.keys()).index(color) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Отображаем кадр в том же окне
        cv2.imshow('Car Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
