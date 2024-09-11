import streamlit as st
import cv2
from fer import FER
import simpleaudio as sa
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from PIL import Image
import tempfile

# Initialize the FER emotion detector
emotion_detector = FER(mtcnn=True)

# Load Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Get the absolute path to the beep sound file
beep_sound_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'beep.wav')

# Load the beep sound
wave_obj = sa.WaveObject.from_wave_file(beep_sound_path)

# Initialize NLP models
qa_model_name = "deepset/roberta-base-squad2"
qa_pipeline = pipeline("question-answering", model=qa_model_name)
tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

# Function to detect faces and emotions
def detect_and_display(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) > 1:
        play_obj = wave_obj.play()
        play_obj.wait_done()
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        
        # Detect emotions
        emotion_results = emotion_detector.detect_emotions(roi_color)
        if emotion_results:
            emotions = emotion_results[0]['emotions']
            dominant_emotion = max(emotions, key=emotions.get)
            cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    
    return frame

# Streamlit Interface
st.sidebar.title("AI Personalized Mock Interview")

# Navigation options
nav_option = st.sidebar.radio("Navigation", ["Home", "Interview", "Contact"])

# Home Section
if nav_option == "Home":
    st.title("Welcome to AI Personalized Mock Interview")
    st.write("""
    This platform provides an AI-powered mock interview experience. Using advanced emotion detection and NLP technologies, we simulate real interview scenarios, analyze your responses, and provide comprehensive feedback to help you improve.
    """)

# Contact Section
elif nav_option == "Contact":
    st.title("Contact Information")
    st.write("""
    For more information, please contact us at:
    - Email: info@mockinterview.ai
    - Phone: +1234567890
    """)

# Interview Section
elif nav_option == "Interview":
    st.title("Mock Interview")
    
    # User information form
    st.sidebar.header("User Information")
    user_name = st.sidebar.text_input("Name")
    user_job_role = st.sidebar.selectbox("Job Role", ["Software Developer (Product Based)", "Software Developer (Service Based)", "Nurse", "Software Manager", "Other"])
    user_age = st.sidebar.number_input("Age", min_value=18, max_value=99)
    user_gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
    user_cv = st.sidebar.file_uploader("Upload your CV", type=["pdf", "doc", "docx"])
    
    # Placeholder for interview
    start_interview = st.button("Start Interview")
    
    if start_interview:
        st.write(f"Welcome {user_name}! Starting your interview for the role of {user_job_role}...")
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Error: Could not open webcam.")
        else:
            stframe = st.empty()
            
            # Interview questions
            questions = [
                "Tell me about yourself.",
                "Why do you want this job?",
                "What are your strengths and weaknesses?",
                "Describe a challenging situation and how you handled it.",
                "Where do you see yourself in 5 years?",
                "Why should we hire you?",
                "What are your salary expectations?",
                "Can you describe a project that you are proud of?",
                "How do you handle conflict at work?",
                "What is your greatest professional achievement?"
            ]
            
            user_answers = []
            
            for question in questions:
                st.write(f"Question: {question}")
                
                # Display webcam feed and detect emotions
                ret, frame = cap.read()
                if ret:
                    frame = detect_and_display(frame)
                    stframe.image(frame, channels="BGR")
                else:
                    st.error("Error: Could not read frame from webcam.")
                
                user_answer = st.text_input(f"Your answer to '{question}'", "")
                user_answers.append((question, user_answer))
                
                st.write("Recording your answer...")
                # Wait for a few seconds
                cv2.waitKey(5000)  # Simulate recording time
                
            cap.release()
            stframe.empty()
            
            st.write("Interview completed!")
            
            # Display interview history and feedback
            st.subheader("Interview History and Feedback")
            for i, (q, a) in enumerate(user_answers):
                st.write(f"Q{i+1}: {q}")
                st.write(f"A{i+1}: {a}")
                
            st.write("Analyzing your responses...")
            
            # Placeholder for feedback (emotion analysis and answer evaluation)
            st.write("Emotion Analysis and Answer Evaluation coming soon...")

# Streamlit main function
if __name__ == '__main__':
    st.set_page_config(page_title="AI Personalized Mock Interview", layout="wide")
    st.sidebar.header("AI Mock Interview")
