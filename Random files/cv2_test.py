import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize Video Capture
cap = cv2.VideoCapture(0)

# Example Emotion Classification Model (Train separately)
# For simplicity, assuming a pre-trained RandomForest model is loaded
emotion_classifier = RandomForestClassifier()
# Load your trained model (e.g., joblib.load("emotion_model.pkl"))

with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_detection.process(image)

        # Convert the image back to BGR for OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detections:
            for detection in results.detections:
                # Draw face detection annotations on the image
                mp_drawing.draw_detection(image, detection)

                # Example: Extract face bounding box and other features
                bbox = detection.location_data.relative_bounding_box
                h, w, c = image.shape
                x, y, width, height = int(bbox.xmin * w), int(bbox.ymin * h), \
                                      int(bbox.width * w), int(bbox.height * h)

                face = image[y:y+height, x:x+width]
                face = cv2.resize(face, (48, 48))  # Resize to model input size
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

                # Flatten face array and predict emotion
                face_flatten = face.flatten().reshape(1, -1)
                emotion_prediction = emotion_classifier.predict(face_flatten)

                # Display the prediction on the image
                emotion_text = f'Emotion: {emotion_prediction[0]}'
                cv2.putText(image, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Emotion Detection', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
