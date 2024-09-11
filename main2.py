import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import os

# Placeholder functions for the ML models
def emotion_detection_model(frame):
    # Dummy implementation - replace with actual model code
    return "Happy", 0.9

def nlp_question_generation(job_role, question_number):
    # Dummy implementation - replace with actual model code
    questions = {
        "Software Engineer": ["What is polymorphism?", "Explain a REST API."],
        "Data Scientist": ["What is a confusion matrix?", "Explain the bias-variance tradeoff."]
    }
    return questions.get(job_role, ["Describe your job role."])[question_number % len(questions.get(job_role, ["Describe your job role."]))]

def nlp_analyze_answer(answer):
    # Dummy implementation - replace with actual model code
    return {"score": 0.8, "feedback": "Good answer"}

def speech_to_text(audio_data):
    # Dummy implementation - replace with actual model code or API call
    return "This is the transcribed text."

# Streamlit UI
def main():
    st.sidebar.header('User Information')
    name = st.sidebar.text_input("Name of the User")
    job_role = st.sidebar.selectbox("Job Role of the User", ["Software Engineer", "Data Scientist", "Product Manager"])
    age = st.sidebar.number_input("Age of the User", min_value=18, max_value=100, value=25)
    gender = st.sidebar.selectbox("Gender of the User", ["Male", "Female", "Other"])

    st.title('AI Powered Mock Interview')
    
    if st.button('Start Interview'):
        st.session_state['interview_started'] = True
        st.session_state['question_number'] = 0
        st.session_state['answers'] = []
        st.session_state['emotions'] = []

    if 'interview_started' in st.session_state and st.session_state['interview_started']:
        st.header("Interview In Progress")
        
        # Placeholder for capturing webcam input
        st.text("Webcam stream and emotion detection placeholder")
        placeholder = st.empty()
        cap = cv2.VideoCapture(0)
        
        question_number = st.session_state['question_number']
        if question_number < 10:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image")
                return
            
            # Display the frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            placeholder.image(img)
            
            # Emotion Detection
            emotion, confidence = emotion_detection_model(frame)
            st.text(f"Detected Emotion: {emotion} (Confidence: {confidence*100:.2f}%)")
            
            # Store emotion data
            st.session_state['emotions'].append((emotion, confidence))
            
            # Question Generation
            question = nlp_question_generation(job_role, question_number)
            st.subheader(f"Question {question_number+1}: {question}")
            
            # Text Answer Input
            answer = st.text_input("Your Answer", key=f"answer_{question_number}")

            # Speech-to-Text Answer Input
            audio_bytes = st.file_uploader("Upload your answer as audio", type=["wav", "mp3"])
            if audio_bytes is not None:
                answer_text = speech_to_text(audio_bytes)
                st.text_area("Transcribed Answer", answer_text, key=f"transcribed_answer_{question_number}")
            
            if st.button("Submit Answer", key=f"submit_{question_number}"):
                final_answer = answer or answer_text
                analysis = nlp_analyze_answer(final_answer)
                st.text(f"Answer Analysis: {analysis['feedback']} (Score: {analysis['score']*100:.2f}%)")
                st.session_state['answers'].append((question, final_answer, analysis))
                st.session_state['question_number'] += 1
        else:
            cap.release()
            st.text("Interview Completed")
            st.session_state['interview_started'] = False

            st.header("Interview History")
            for i, (question, answer, analysis) in enumerate(st.session_state['answers']):
                st.text(f"Q{i+1}: {question}")
                st.text(f"A{i+1}: {answer}")
                st.text(f"Q{i+1} Score: {analysis['score']*100:.2f}% - {analysis['feedback']}")
                
            st.header("Feedback Report")
            for i, (question, answer, analysis) in enumerate(st.session_state['answers']):
                st.text(f"Q{i+1} Score: {analysis['score']*100:.2f}% - {analysis['feedback']}")
                
            # Download report as text file
            if st.button("Download Report"):
                report_path = "report.txt"
                with open(report_path, "w") as f:
                    for i, (question, answer, analysis) in enumerate(st.session_state['answers']):
                        f.write(f"Q{i+1}: {question}\n")
                        f.write(f"A{i+1}: {answer}\n")
                        f.write(f"Q{i+1} Score: {analysis['score']*100:.2f}% - {analysis['feedback']}\n\n")
                with open(report_path, "rb") as file:
                    btn = st.download_button(
                        label="Download Report",
                        data=file,
                        file_name="report.txt",
                        mime="text/plain"
                    )
                if os.path.exists(report_path):
                    os.remove(report_path)

if __name__ == "__main__":
    main()
