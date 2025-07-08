import cv2
import time
from ultralytics import YOLO

# --- Configuration ---
# IMPORTANT: If your camera does not require a username and password, 
# the URL format is typically rtsp://<IP>:<PORT>/<PATH>
# The placeholder is a common format for Hikvision/Amcrest cameras.
RTSP_URL = "rtsp://dnakhla%40gmail.com:vi6Z39oCPz-iYmZ@192.168.86.42:554/h264Preview_01_main"

# --- Initialization ---
print("Connecting to RTSP stream...")
# Use FFMPEG backend for better RTSP support
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    print("Please check the following:")
    print("1. The RTSP URL is correct (user, password, IP, port, channel).")
    print("2. The camera is online and RTSP is enabled in its web UI.")
    print("3. You have installed FFmpeg on your system (`brew install ffmpeg`).")
    exit()

print("Stream opened successfully.")

# Load the YOLOv8n (nano) model. It will be downloaded on the first run.
# This is a small, fast model suitable for real-time CPU inference.
# For maximum performance on Apple Silicon, export this to CoreML.
print("Loading YOLOv8 model...")
model = YOLO("yolov8n.mlpackage")
print("Model loaded.")

def draw_detections(frame, boxes):
    """Draws bounding boxes and labels on the video frame."""
    for b in boxes:
        # Extract class name and confidence score
        class_id = int(b.cls.item())
        confidence = b.conf.item()
        label = f"{model.names[class_id]}: {confidence:.2f}"

        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, b.xyxy[0])

        # Draw rectangle and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# --- Main Loop ---
print("Starting inference loop. Press 'Esc' in the window to quit.")
while True:
    # Read a frame from the stream
    ok, frame = cap.read()
    if not ok:
        print("Warning: Failed to grab frame. The stream may have ended.")
        break

    t0 = time.time()

    # Run inference on the current frame
    # `imgsz=640` standardizes the input image size for the model
    # `verbose=False` keeps the console clean
    results = model(frame, imgsz=640, verbose=False)

    # The first element in results contains the detection boxes
    boxes = results[0].boxes
    draw_detections(frame, boxes)

    # --- Console Output ---
    # Build a list of detected objects for printing
    detections_text = []
    if len(boxes) > 0:
        for b in boxes:
            class_id = int(b.cls.item())
            confidence = b.conf.item()
            detections_text.append(f"{model.names[class_id]} ({confidence:.2f})")

    # Calculate FPS
    fps = 1 / (time.time() - t0)

    # Print FPS and the list of detected objects
    print(f"FPS: {fps:.1f} | Detections: {', '.join(detections_text) if detections_text else 'None'}")

    # Display the annotated frame
    cv2.imshow("DoorbellAI - Live Inference", frame)

    # Exit loop if 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        print("Escape key pressed. Exiting...")
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("Resources released.")
