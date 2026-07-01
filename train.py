from ultralytics import YOLO

model = YOLO('yolo11n.pt')
model.train(
    data='yolo_config.yaml',
    epochs=150,
    imgsz=640,
    batch=16
)
