import cv2
from fer import FER
import simpleaudio as sa
import os

# Initialize the FER emotion detector
emotion_detector = FER(mtcnn=True)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Get the absolute path to the beep sound file
beep_sound_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'beep.wav')

# Load the beep sound
wave_obj = sa.WaveObject.from_wave_file(beep_sound_path)

def detect_and_display_emotions(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) > 1:
        play_obj = wave_obj.play()
        play_obj.wait_done()
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_color = frame[y:y+h, x:x+w]
        
        # Detect emotion
        emotion_results = emotion_detector.detect_emotions(roi_color)
        if emotion_results:
            emotions = emotion_results[0]['emotions']
            dominant_emotion = max(emotions, key=emotions.get)
            cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    
    return frame

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect and display emotions in the frame
    frame = detect_and_display_emotions(frame)
    
    # Display the frame
    cv2.imshow('Emotion Detection with Beep', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
