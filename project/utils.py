import cv2
import numpy as np

# Глобальные переменные
regions = {'red': [], 'blue': [], 'green': [], 'black': []}
car_count = {'red': 0, 'blue': 0, 'green': 0, 'black': 0}
color_dict = {'red': (0, 0, 255), 'blue': (255, 0, 0), 'green': (0, 255, 0), 'black': (0, 0, 0)}
scale_factor = 1.5  # Масштабирование

def draw_regions(image):
    """Рисует зоны на изображении."""
    for color, pts in regions.items():
        if len(pts) > 1:
            cv2.polylines(image, [np.array(pts)], isClosed=True, color=color_dict[color], thickness=2)

def is_inside_zone(center, zone):
    """Проверяет, находится ли точка внутри зоны."""
    if len(zone) == 0:
        return False
    return cv2.pointPolygonTest(np.array(zone, dtype=np.int32), center, False) >= 0
