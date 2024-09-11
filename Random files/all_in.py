import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np

# Initialize Mediapipe for face and hand detection
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# Load pre-trained models for eye detection and emotion detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
emotion_model = load_model('emotion_detection_model.h5')  # Updated to use your model

# Define a function to preprocess the face for emotion detection
def preprocess_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48, 48))
    face = face.astype("float") / 255.0
    face = np.reshape(face, (1, 48, 48, 1))
    return face

# Initialize video capture
cap = cv2.VideoCapture(0)
multi_face_warnings = 0  # Counter for multiple face detection warnings

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
     mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        # Process the image and detect face and hands
        face_results = face_mesh.process(frame_rgb)
        hand_results = hands.process(frame_rgb)

        # Convert back to BGR for drawing
        frame_rgb.flags.writeable = True
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Face detection and emotion analysis
        if face_results.multi_face_landmarks:
            num_faces = len(face_results.multi_face_landmarks)
            if num_faces > 1:
                multi_face_warnings += 1
                print(f"Warning: Multiple faces detected! ({multi_face_warnings}/5)")

                if multi_face_warnings >= 5:
                    print("Too many warnings! Terminating the program.")
                    break
            else:
                multi_face_warnings = 0  # Reset warnings if only one face is detected

            for face_landmarks in face_results.multi_face_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image=frame_bgr,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_tesselation_style())

                # Extract the face ROI
                face_bbox = mp.solutions.drawing_utils._normalized_to_pixel_coordinates(
                    face_landmarks.landmark[10].x, face_landmarks.landmark[10].y,
                    frame_bgr.shape[1], frame_bgr.shape[0]
                )

                if face_bbox:
                    x, y = face_bbox
                    face_roi = frame_bgr[y:y + 200, x:x + 200]  # Adjust ROI dimensions as needed

                    # Detect eyes in the face ROI
                    eyes = eye_cascade.detectMultiScale(face_roi)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(face_roi, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                    # Detect emotions
                    preprocessed_face = preprocess_face(face_roi)
                    emotion_prediction = emotion_model.predict(preprocessed_face)
                    emotion_label = np.argmax(emotion_prediction)
                    emotion_text = f"Emotion: {emotion_label}"
                    cv2.putText(frame_bgr, emotion_text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Hand detection
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image=frame_bgr,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS)

        # Display the frame with all detections
        cv2.imshow('Real-Time Monitoring', frame_bgr)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
