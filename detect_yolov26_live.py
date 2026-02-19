from picamera2 import Picamera2
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time
import os
import csv
from datetime import datetime

os.chdir("/home/pi")
time.sleep(5)

# ================= CONFIG =================
MODEL_PATH = "/home/pi/yolov26_640/best_full_integer_quant.tflite"
INPUT_SIZE = 640
CONF_THRESHOLD = 0.80
FRAME_SKIP = 2
CLIP_DURATION = 5
SAVE_DIR = "detection_logs"
FULL_FPS = 20
# ==========================================

# Create folders
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(f"{SAVE_DIR}/clips", exist_ok=True)

CSV_PATH = f"{SAVE_DIR}/detections.csv"

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "clip_name"])

# ================= LOAD MODEL =================
interpreter = tflite.Interpreter(
    model_path=MODEL_PATH,
    num_threads=4
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_scale, input_zero = input_details[0]['quantization']
output_scale, output_zero = output_details[0]['quantization']

# ================= CAMERA =================
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (480, 360)}
)
picam2.configure(config)
picam2.start()

# ================= FULL DRIVE RECORDING =================
drive_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
full_video_path = f"{SAVE_DIR}/full_drive_{drive_timestamp}.mp4"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
full_video_writer = cv2.VideoWriter(
    full_video_path,
    fourcc,
    FULL_FPS,
    (480, 360)
)

print(f"🚗 Full Drive Recording: {full_video_path}")

# ================= CLIP RECORDING =================
recording = False
record_start_time = None
clip_writer = None

frame_count = 0

print("🚦 YOLOv26 Live Detection Started")

while True:
    frame = picam2.capture_array()
    frame = np.ascontiguousarray(frame[:, :, :3])
    orig_h, orig_w = frame.shape[:2]

    display_frame = frame.copy()

    # Always record full drive
    full_video_writer.write(display_frame)

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        cv2.imshow("YOLOv26 Live Detection", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    start = time.time()

    # ================= PREPROCESS =================
    resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    input_data = rgb.astype(np.float32) / 255.0
    input_data = input_data / input_scale + input_zero
    input_data = np.clip(input_data, -128, 127).astype(np.int8)
    input_data = np.expand_dims(input_data, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # ================= POSTPROCESS =================
    output = interpreter.get_tensor(output_details[0]['index'])
    output = (output.astype(np.float32) - output_zero) * output_scale
    detections = output[0]

    anomaly_detected = False

    for det in detections:
        if len(det) < 6:
            continue

        x1, y1, x2, y2, obj, cls_conf = det
        conf = cls_conf  # Correct column (as discovered)

        if conf < CONF_THRESHOLD:
            continue

        anomaly_detected = True

        # Normalized coords → pixel coords
        x1 = int(x1 * orig_w)
        y1 = int(y1 * orig_h)
        x2 = int(x2 * orig_w)
        y2 = int(y2 * orig_h)

        x1 = max(0, min(x1, orig_w - 1))
        y1 = max(0, min(y1, orig_h - 1))
        x2 = max(0, min(x2, orig_w - 1))
        y2 = max(0, min(y2, orig_h - 1))

        label = f"Anomaly {conf:.2f}"

        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(display_frame, label,
                    (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,255,0), 2)

    # ================= TRIGGER CLIP =================
    if anomaly_detected and not recording:
        recording = True
        record_start_time = time.time()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clip_name = f"clip_{timestamp}.mp4"
        clip_path = f"{SAVE_DIR}/clips/{clip_name}"

        clip_writer = cv2.VideoWriter(
            clip_path,
            fourcc,
            FULL_FPS,
            (orig_w, orig_h)
        )

        with open(CSV_PATH, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, clip_name])

        print(f"🚨 Anomaly detected → Recording {clip_name}")

    # Continue recording clip
    if recording:
        clip_writer.write(display_frame)

        if time.time() - record_start_time > CLIP_DURATION:
            recording = False
            clip_writer.release()
            print("✅ Clip saved")

    # ================= FPS DISPLAY =================
    fps = 1 / (time.time() - start)
    cv2.putText(display_frame, f"FPS: {fps:.2f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0,0,255), 2)

    cv2.imshow("YOLOv26 Live Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================= CLEANUP =================
print("Stopping...")

full_video_writer.release()

if clip_writer:
    clip_writer.release()

picam2.stop()
cv2.destroyAllWindows()
