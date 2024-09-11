import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from transformers import pipeline

# Initialize Mediapipe for Face Detection and Face Mesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize Sentiment Analysis Pipeline
sentiment_analysis = pipeline("sentiment-analysis")

# Example Emotion Classification Model (Train separately)
emotion_classifier = RandomForestClassifier()
# Load your trained model, e.g., joblib.load("emotion_model.pkl")

# Initialize Video Capture
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection, \
     mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
     mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the image for face detection and hand detection
        face_detection_results = face_detection.process(image)
        face_mesh_results = face_mesh.process(image)
        hand_results = hands.process(image)

        # Convert the image back to BGR for OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Face Detection and Emotion Classification
        if face_detection_results.detections:
            for detection in face_detection_results.detections:
                mp_drawing.draw_detection(image, detection)

                # Extract face bounding box
                bbox = detection.location_data.relative_bounding_box
                h, w, c = image.shape
                x, y, width, height = int(bbox.xmin * w), int(bbox.ymin * h), \
                                      int(bbox.width * w), int(bbox.height * h)

                face = image[y:y + height, x:x + width]
                face = cv2.resize(face, (48, 48))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                # Flatten face array and predict emotion
                face_flatten = face.flatten().reshape(1, -1)
                emotion_prediction = emotion_classifier.predict(face_flatten)

                # Display the prediction on the image
                emotion_text = f'Emotion: {emotion_prediction[0]}'
                cv2.putText(image, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Real-Time Monitoring for Eye Contact, Gestures, and Non-verbal Cues
        if face_mesh_results.multi_face_landmarks:
            for face_landmarks in face_mesh_results.multi_face_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_tesselation_style())

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Interview Analysis', image)

        # Example: Running Sentiment Analysis on Simulated User Responses
        if cv2.waitKey(5) & 0xFF == ord('r'):  # Press 'r' to run sentiment analysis
            user_responses = [
                "I am very excited about this job opportunity.",
                "I feel a bit nervous about the technical questions.",
                "I think I did well in the interview, but I'm not sure."
            ]

            for response in user_responses:
                sentiment = sentiment_analysis(response)[0]
                print(f"Response: {response}")
                print(f"Sentiment: {sentiment['label']}, Confidence: {sentiment['score']}\n")

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break

cap.release()
cv2.destroyAllWindows()
