import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load pre-trained models for face detection and emotion detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # OpenCV face detector

# Load pre-trained emotion detection model from local file
# Replace 'path_to_your_model' with the actual path to the downloaded model file
emotion_model_path = 'emotion_detection_model.h5'
emotion_model = tf.keras.models.load_model(emotion_model_path)

# Initialize Mediapipe for hand detection
mp_hands = mp.solutions.hands

# Initialize video capture
cap = cv2.VideoCapture(0)
multi_face_warnings = 0  # Counter for multiple face detection warnings

# Function to preprocess face for emotion detection
def preprocess_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48, 48))
    face = face.astype("float") / 255.0
    face = np.reshape(face, (1, 48, 48, 1))
    return face

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        # Detect faces
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Count and handle multiple faces
        num_faces = len(faces)
        if num_faces > 1:
            multi_face_warnings += 1
            print(f"Warning: Multiple faces detected! ({multi_face_warnings}/5)")

            if multi_face_warnings >= 5:
                print("Too many warnings! Terminating the program.")
                break
        else:
            multi_face_warnings = 0  # Reset warnings if only one face is detected

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]

            # Detect emotions
            preprocessed_face = preprocess_face(face_roi)
            emotion_prediction = emotion_model.predict(preprocessed_face)
            emotion_label = np.argmax(emotion_prediction)
            emotion_text = f"Emotion: {emotion_label}"
            cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Draw rectangle around faces
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Detect hands
        hand_results = hands.process(frame_rgb)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image=frame,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS)

        # Display the frame with all detections
        cv2.imshow('Real-Time Monitoring', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
