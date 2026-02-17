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
MODEL_PATH = "/home/pi/nano_320/best_full_integer_quant.tflite"
INPUT_SIZE = 320
CONF_THRESHOLD = 0.5
FRAME_SKIP = 2
CLIP_DURATION = 5  # seconds
SAVE_DIR = "detection_logs"
# ==========================================

# Create folders
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(f"{SAVE_DIR}/clips", exist_ok=True)

CSV_PATH = f"{SAVE_DIR}/detections.csv"

# Create CSV if not exists
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "clip_name"])

# Load INT8 model
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

recording = False
record_start_time = None
video_writer = None

frame_count = 0

print("🚗 Road Test Mode Started")

while True:
    frame = picam2.capture_array()
    frame = np.ascontiguousarray(frame[:, :, :3])

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    start = time.time()

    # Resize for model
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

            x1, y1, x2, y2 = det[:4]
            x1 = int(x1 * frame.shape[1] / INPUT_SIZE)
            y1 = int(y1 * frame.shape[0] / INPUT_SIZE)
            x2 = int(x2 * frame.shape[1] / INPUT_SIZE)
            y2 = int(y2 * frame.shape[0] / INPUT_SIZE)

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    # 🚨 Trigger recording
    if anomaly_detected and not recording:
        recording = True
        record_start_time = time.time()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clip_name = f"clip_{timestamp}.mp4"
        clip_path = f"{SAVE_DIR}/clips/{clip_name}"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            clip_path,
            fourcc,
            20,
            (frame.shape[1], frame.shape[0])
        )

        # Save to CSV
        with open(CSV_PATH, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, clip_name])

        print(f"🚨 Anomaly detected! Recording {clip_name}")

    # Continue recording
    if recording:
        video_writer.write(frame)

        if time.time() - record_start_time > CLIP_DURATION:
            recording = False
            video_writer.release()
            print("✅ Clip saved.")

    fps = 1 / (time.time() - start)
    cv2.putText(frame, f"FPS: {fps:.2f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0,0,255), 2)

    cv2.imshow("Road Test Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if video_writer:
    video_writer.release()

cv2.destroyAllWindows()
