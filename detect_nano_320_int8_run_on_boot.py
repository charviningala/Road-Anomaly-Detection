from picamera2 import Picamera2
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time
import os
import csv
from datetime import datetime

# ================= CONFIG =================
MODEL_PATH = "/home/pi/nano_320/best_full_integer_quant.tflite"
INPUT_SIZE = 320
CONF_THRESHOLD = 0.5
FRAME_SKIP = 2
CLIP_DURATION = 5  # seconds
SAVE_DIR = "/home/pi/detection_logs"
FULL_VIDEO_FPS = 20
# ==========================================

time.sleep(10)  # Give camera + system time to initialize

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(f"{SAVE_DIR}/clips", exist_ok=True)

CSV_PATH = f"{SAVE_DIR}/detections.csv"

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "clip_name"])

# Load TFLite INT8 model
interpreter = tflite.Interpreter(
    model_path=MODEL_PATH,
    num_threads=4
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
scale, zero_point = input_details[0]['quantization']

# Camera setup
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (480, 360)}
)
picam2.configure(config)
picam2.start()

# 🎥 FULL SESSION VIDEO WRITER
session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
full_video_path = f"{SAVE_DIR}/full_session_{session_timestamp}.avi"

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
full_video_writer = cv2.VideoWriter(
    full_video_path,
    fourcc,
    FULL_VIDEO_FPS,
    (480, 360)
)

if not full_video_write.isOpened():
	raise RuntimeError("Full Session writer failed")

print("🚗 Boot Detection Started")

recording_clip = False
clip_writer = None
record_start_time = None

frame_count = 0

try:
    while True:
        frame = picam2.capture_array()
        frame = np.ascontiguousarray(frame[:, :, :3])

        # Always write full session video
        full_video_writer.write(frame)

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        # -------- Inference --------
        resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        input_data = resized.astype(np.float32) / 255.0
        input_data = input_data / scale + zero_point
        input_data = input_data.astype(np.int8)
        input_data = np.expand_dims(input_data, axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        detections = np.squeeze(output_data)

        anomaly_detected = False

        for det in detections[:20]:
            score = det[4]
            if score > CONF_THRESHOLD:
                anomaly_detected = True

        # 🚨 START CLIP RECORDING
        if anomaly_detected and not recording_clip:
            recording_clip = True
            record_start_time = time.time()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            clip_name = f"clip_{timestamp}.avi"
            clip_path = f"{SAVE_DIR}/clips/{clip_name}"

            clip_writer = cv2.VideoWriter(
                clip_path,
                fourcc,
                FULL_VIDEO_FPS,
                (480, 360)
            )

            with open(CSV_PATH, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, clip_name])

            print(f"🚨 Anomaly detected: {clip_name}")

        # Continue writing clip
        if recording_clip:
            clip_writer.write(frame)

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
