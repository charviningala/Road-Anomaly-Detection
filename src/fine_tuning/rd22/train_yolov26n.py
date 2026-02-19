from ultralytics import YOLO

PRETRAINED_MODEL = "/home/saber/GitHub/road_anomaly_detection/runs/three_country_training/road_defect_binary/yolov26s2/weights/best.pt"

model = YOLO(PRETRAINED_MODEL)


save_dir = "../../../runs/yolov26s2_rdd2022_2class"

from pathlib import Path
print(Path("../../../data/rdd2class_yolo").resolve())

model.train(
    data=Path("/home/saber/GitHub/road_anomaly_detection/data/rdd2class_yolo/rdd2class.yaml"),
    epochs=70,
    imgsz=640,
    batch=16,
    freeze=10,
    name="yolov26s2_rdd2022_2class"
)