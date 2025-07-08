from ultralytics import YOLO

print("Loading base YOLOv8 model...")
# This will download the yolov8n.pt file if it doesn't exist
model = YOLO("yolov8n.pt")

print("Exporting model to CoreML format...")
# Exports the model to yolov8n.mlpackage
# This format is optimized for Apple hardware (GPU / Neural Engine)
model.export(format="coreml")

print("Export complete. You can now use 'yolov8n.mlpackage' in the main script.")
