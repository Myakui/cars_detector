import cv2
from zone_selection import select_zones, draw_regions
from car_detection import load_yolo_model, detect_cars
from utils import is_inside_zone, car_count, regions, scale_factor

def main():
    model = load_yolo_model()

    print("Укажите путь до видео")
    path = input()
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видео.")
        return

    # Окно для выбора зон и детекции машин
    cv2.namedWindow('Car Detection', cv2.WINDOW_NORMAL)
    
    # Выбор зон
    select_zones(cap, 'Car Detection')

    # Основной цикл для детекции машин
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

        # Рисуем зоны
        draw_regions(frame)

        # Детектируем машины
        detections = detect_cars(frame, model)

        for (x, y, w, h) in detections:
            center = (x + w // 2, y + h // 2)
            for color in regions:
                if is_inside_zone(center, regions[color]):
                    car_count[color] += 1
                    break

            # Рисуем прямоугольники вокруг машин
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Отображаем счетчики машин
        for color, count in car_count.items():
            cv2.putText(frame, f'{color.capitalize()}: {count}', 
                        (10, 30 + list(car_count.keys()).index(color) * 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Отображаем кадр
        cv2.imshow('Car Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
