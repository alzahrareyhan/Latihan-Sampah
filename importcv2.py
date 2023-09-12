
from ultralytics import YOLO
import cv2

# Load the YOLOv5 model
model = YOLO("yolov8n.pt")

# Perform object detection on source (webcam or video)
results = model.predict(source="0", show=True)

# Get detections for "sampah" class (assuming class index is known, adjust if needed)
sampah_results = results.pred[0][results.pred[0][:, 5] == sampah_class_index]

# Render and show the detections for "sampah" class
sampah_results.render()  # Render the detected objects on the image
sampah_results.show()    # Show the image with detections
print(results)

# Keep the window open until a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
