import streamlit as st
import cv2
import numpy as np
from PIL import Image
import speech_recognition as sr
import torch
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
from torchvision import models, transforms

# Load models
def load_emotion_detection_model():
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 7)  # Assuming 7 emotion classes
    model.eval()
    return model

def load_nlp_model():
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    return model, tokenizer

def emotion_detection_model(frame, model):
    # Preprocess frame
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(frame).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    emotion = emotions[predicted.item()]
    confidence = torch.nn.functional.softmax(outputs, dim=1).max().item()
    return emotion, confidence

def nlp_question_generation(job_role, question_number, nlp_model, nlp_tokenizer):
    questions = {
        "Software Engineer": ["What is polymorphism?", "Explain a REST API."],
        "Data Scientist": ["What is a confusion matrix?", "Explain the bias-variance tradeoff."]
    }
    return questions.get(job_role, ["Describe your job role."])[question_number % len(questions.get(job_role, ["Describe your job role."]))]

def nlp_analyze_answer(answer, nlp_model, nlp_tokenizer):
    inputs = nlp_tokenizer.encode("analyze: " + answer, return_tensors="pt", max_length=512, truncation=True)
    outputs = nlp_model.generate(inputs, max_length=150, num_beams=2, early_stopping=True)
    analysis = nlp_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"score": 0.8, "feedback": analysis}  # Placeholder score and feedback

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
        st.session_state['emotion_model'] = load_emotion_detection_model()
        st.session_state['nlp_model'], st.session_state['nlp_tokenizer'] = load_nlp_model()

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
            emotion, confidence = emotion_detection_model(frame, st.session_state['emotion_model'])
            st.text(f"Detected Emotion: {emotion} (Confidence: {confidence*100:.2f}%)")
            
            # Store emotion data
            st.session_state['emotions'].append((emotion, confidence))
            
            # Question Generation
            question = nlp_question_generation(job_role, question_number, st.session_state['nlp_model'], st.session_state['nlp_tokenizer'])
            st.subheader(f"Question {question_number+1}: {question}")
            
            # Speech-to-Text Answer Input
            with st.spinner("Listening..."):
                r = sr.Recognizer()
                with sr.Microphone() as source:
                    st.write("Speak your answer...")
                    audio = r.listen(source)
                try:
                    answer_text = r.recognize_google(audio)
                    st.text_area("Transcribed Answer", answer_text, key=f"transcribed_answer_{question_number}")
                except sr.UnknownValueError:
                    st.error("Sorry, could not understand audio")
                except sr.RequestError as e:
                    st.error(f"Could not request results; {e}")

            if st.button("Submit Answer", key=f"submit_{question_number}"):
                analysis = nlp_analyze_answer(answer_text, st.session_state['nlp_model'], st.session_state['nlp_tokenizer'])
                st.text(f"Answer Analysis: {analysis['feedback']} (Score: {analysis['score']*100:.2f}%)")
                st.session_state['answers'].append((question, answer_text, analysis))
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
                with st.spinner("Generating report..."):
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
