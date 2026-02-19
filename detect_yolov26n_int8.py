from picamera2 import Picamera2
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time
import os
import csv
from datetime import datetime
import pytz

# ================= CONFIG =================
MODEL_PATH = "/home/pi/yolov26n_quant/best_full_integer_quant.tflite"
INPUT_SIZE = 640
CONF_THRESHOLD = 0.60
FRAME_SKIP = 2
CLIP_DURATION = 5
SAVE_DIR = "/home/pi/detection_logs"
FULL_VIDEO_FPS = 20
# ==========================================

time.sleep(10)

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(f"{SAVE_DIR}/clips", exist_ok=True)

CSV_PATH = f"{SAVE_DIR}/detections.csv"

# IST timezone
IST = pytz.timezone("Asia/Kolkata")

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp_utc",
            "timestamp_ist",
            "detection_type",
            "clip_name"
        ])

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

# ================= FULL VIDEO =================
session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
full_video_path = f"{SAVE_DIR}/full_session_{session_timestamp}.avi"

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
full_video_writer = cv2.VideoWriter(
    full_video_path,
    fourcc,
    FULL_VIDEO_FPS,
    (480, 360)
)

if not full_video_writer.isOpened():
    raise RuntimeError("Full Session writer failed")

print("🚗 YOLOv26 Road Anomaly Detection Started")

recording_clip = False
clip_writer = None
record_start_time = None

frame_count = 0

try:
    while True:
        frame = picam2.capture_array()
        frame = np.ascontiguousarray(frame[:, :, :3])
        orig_h, orig_w = frame.shape[:2]

        display_frame = frame.copy()

        frame_count += 1

        detected_types = []

        if frame_count % FRAME_SKIP == 0:

            # -------- PREPROCESS --------
            resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            input_data = rgb.astype(np.float32) / 255.0
            input_data = input_data / input_scale + input_zero
            input_data = np.clip(input_data, -128, 127).astype(np.int8)
            input_data = np.expand_dims(input_data, axis=0)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            # -------- POSTPROCESS --------
            output = interpreter.get_tensor(output_details[0]['index'])
            output = (output.astype(np.float32) - output_zero) * output_scale
            detections = output[0]

            anomaly_detected = False

            for det in detections:
                if len(det) < 6:
                    continue

                x1, y1, x2, y2, obj, cls_conf = det
                conf = obj * cls_conf

                if conf < CONF_THRESHOLD:
                    continue

                anomaly_detected = True
                detection_type = "Anomaly"  # You can customize if multi-class
                detected_types.append(detection_type)

                # Normalized → pixel
                x1 = int(x1 * orig_w)
                y1 = int(y1 * orig_h)
                x2 = int(x2 * orig_w)
                y2 = int(y2 * orig_h)

                x1 = max(0, min(x1, orig_w - 1))
                y1 = max(0, min(y1, orig_h - 1))
                x2 = max(0, min(x2, orig_w - 1))
                y2 = max(0, min(y2, orig_h - 1))

                label = f"{detection_type} {conf:.2f}"

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(display_frame, label,
                            (x1, max(20, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,255,0), 2)

            # 🚨 START CLIP RECORDING
            if anomaly_detected and not recording_clip:
                recording_clip = True
                record_start_time = time.time()

                utc_now = datetime.utcnow()
                ist_now = datetime.now(IST)

                timestamp_utc = utc_now.strftime("%Y-%m-%d %H:%M:%S")
                timestamp_ist = ist_now.strftime("%Y-%m-%d %H:%M:%S")

                clip_name = f"clip_{ist_now.strftime('%Y%m%d_%H%M%S')}.avi"
                clip_path = f"{SAVE_DIR}/clips/{clip_name}"

                clip_writer = cv2.VideoWriter(
                    clip_path,
                    fourcc,
                    FULL_VIDEO_FPS,
                    (480, 360)
                )

                with open(CSV_PATH, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        timestamp_utc,
                        timestamp_ist,
                        ",".join(set(detected_types)),
                        clip_name
                    ])

                print(f"🚨 Anomaly detected: {clip_name}")

        full_video_writer.write(display_frame)

        if recording_clip:
            clip_writer.write(display_frame)

            if time.time() - record_start_time > CLIP_DURATION:
                recording_clip = False
                clip_writer.release()
                print("✅ Clip saved")

except Exception as e:
    print("❌ Fatal error:", e)

finally:
    if clip_writer:
        clip_writer.release()
    if full_video_writer:
        full_video_writer.release()
    picam2.stop()
