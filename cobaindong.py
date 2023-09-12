import cv2
import torch
from pathlib import Path

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov8', 'custom', path_or_model='yolov8s.pt')

# Open a video stream
cap = cv2.VideoCapture(0)  # 0 for default camera, or provide the video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Process the results
    detected_objects = results.pred[0]
    for det in detected_objects:
        label = model.names[int(det[5])]
        confidence = det[4].item()
        bbox = det[:4].cpu().numpy()
        x1, y1, x2, y2 = bbox

        if confidence > 0.5:  # Adjust this threshold as needed
            color = (0, 255, 0)  # Green color for bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Trash Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
