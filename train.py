from ultralytics import YOLO

model = YOLO('yolo11n.pt')
model.train(
    #data="./path_to_data_yaml.yaml",
    data='/home/aisd_user12/yolo_config.yaml',
    epochs=100,
    imgsz=640,
    batch=16
) 
