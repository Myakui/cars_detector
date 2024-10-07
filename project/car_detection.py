from ultralytics import YOLO

def load_yolo_model():
    """Загрузка модели YOLOv8."""
    model = YOLO('yolov8n.pt')
    return model

def detect_cars(frame, model):
    """Детекция машин на кадре."""
    results = model(frame)
    detections = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        for box, cls, conf in zip(boxes, classes, confidences):
            if int(cls) == 2 and conf > 0.5:  # ID 2 - машины
                x1, y1, x2, y2 = map(int, box[:4])
                detections.append((x1, y1, x2 - x1, y2 - y1))
    return detections
