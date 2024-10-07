import cv2
from utils import regions, color_dict, draw_regions

drawing = False
current_color_idx = 0

def select_region(event, x, y, flags, param):
    """Обработка кликов мыши для выбора зон."""
    global drawing, current_color_idx
    current_color = list(regions.keys())[current_color_idx]

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        if len(regions[current_color]) < 4:  # Только 4 точки
            regions[current_color].append((x, y))

        if len(regions[current_color]) == 4:
            current_color_idx = (current_color_idx + 1) % len(regions)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def select_zones(cap, window_name):
    """Выбор зон для подсчета машин."""
    cv2.setMouseCallback(window_name, select_region)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: Не удалось прочитать кадр.")
            break

        frame = cv2.resize(frame, (0, 0), fx=1.5, fy=1.5)
        draw_regions(frame)

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
