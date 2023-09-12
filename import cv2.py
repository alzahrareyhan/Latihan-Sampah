import cv2
import numpy as np

# Load pre-trained model for object detection (You might need to install this model first)
net = cv2.dnn.readNet('path_to_your_model_weights', 'path_to_your_model_config')

# Load class labels (if applicable)
classes = []
with open('path_to_your_class_labels', 'r') as f:
    classes = f.read().splitlines()

# Initialize camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while True:
    ret, frame = cap.read()  # Read frame from camera
    if not ret:
        break

    # Prepare the input image for the model
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Perform object detection
    detections = net.forward()

    # Process the detections
    for detection in detections[0]:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:  # You can adjust this threshold
            center_x = int(detection[0] * frame.shape[1])
            center_y = int(detection[1] * frame.shape[0])
            width = int(detection[2] * frame.shape[1])
            height = int(detection[3] * frame.shape[0])

            x = int(center_x - width / 2)
            y = int(center_y - height / 2)

            # Draw bounding box and label
            color = (0, 255, 0)  # Green color for the bounding box
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
            cv2.putText(frame, classes[class_id], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
