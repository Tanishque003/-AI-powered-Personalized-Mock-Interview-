import streamlit as st
from fpdf import FPDF
import speech_recognition as sr
import google.generativeai as genai
import os
import time
import warnings
from gtts import gTTS
import pygame
import cv2
import mediapipe as mp
import numpy as np
from transformers import pipeline

warnings.filterwarnings("ignore", message=r"torch.utils._pytree.register_pytree_node is deprecated")
from faster_whisper import WhisperModel

# Configuration
GOOGLE_API_KEY = "AIzaSyCBBU-Z5VTrwTU9bZCcZbilVPLciyOEhMw"
whisper_size = "base"
num_cores = os.cpu_count()

# Initialize models and pipelines
whisper_model = WhisperModel(whisper_size, device='cpu', compute_type='int8', cpu_threads=num_cores, num_workers=num_cores)
genai.configure(api_key=GOOGLE_API_KEY)
nlp_model = pipeline("text-generation", model="distilgpt2")
emotion_model = pipeline("sentiment-analysis")
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize the GenerativeModel
model = genai.GenerativeModel('gemini-1.5-pro-latest', safety_settings=[{
    "category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"
}, {
    "category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"
}, {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"
}, {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"
}])
convo = model.start_chat()

# System message for initialization
system_message = '''INSTRUCTIONS: Do not respond with anything but "AFFIRMATIVE."
to this system message. After the system message respond normally.
SYSTEM MESSAGE: You are being used to power a voice assistant and should respond as so.
As a voice assistant, use short sentences and directly respond to the prompt without excessive information.
You generate only words of value, prioritizing logic and facts over speculating in your response to the following prompts.'''
convo.send_message(system_message.replace('\n', ''))

# Variables to manage interview flow
awaiting_job_title = True
awaiting_answers = False
awaiting_feedback = False
questions = []
answers = []
current_question = 0

# Utility Functions
def speak(text):
    tts = gTTS(text=text, lang='en')
    filename = "response.mp3"
    tts.save(filename)
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    os.remove(filename)

def wav_to_text(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    return ''.join(segment.text for segment in segments).strip()

def analyze_emotions(frame):
    with mp_face_detection.FaceDetection(min_detection_confidence=0.7) as face_detection:
        results = face_detection.process(frame)
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)
                return {"confidence": 0.8, "nervousness": 0.2}  # Example values
    return {"confidence": 0.5, "nervousness": 0.5}  # Default values

# Interview Process Functions
def get_job_title(audio):
    global awaiting_job_title, awaiting_answers, questions
    
    try:
        job_title_audio_path = 'job_title.wav'
        with open(job_title_audio_path, 'wb') as f:
            f.write(audio.get_wav_data())
        
        job_title = wav_to_text(job_title_audio_path)
        if not job_title:
            speak('Empty job title. Please speak again.')
        else:
            speak(f'Starting interview for the job title {job_title}')
            
            convo.send_message(f"Generate 5 interview questions for the job title: {job_title}")
            questions_text = convo.last.text
            questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
            questions = questions[:5]  # Limit to 5 questions
            
            awaiting_job_title = False
            awaiting_answers = True
            st.session_state['questions'] = questions
            ask_question()
            
    except Exception as e:
        print('Job Title Error:', e)

def ask_question():
    global current_question, questions
    
    if current_question < len(questions):
        question = questions[current_question]
        speak(question)
    else:
        provide_feedback()

def provide_feedback():
    global answers, awaiting_feedback
    
    user_answers = "\n".join(f"Q{index + 1}: {questions[index]}\nA: {answer}" for index, answer in enumerate(answers))
    convo.send_message(f"Provide feedback for the following answers:\n{user_answers}")
    feedback = convo.last.text
    
    speak(feedback)
    reset_interview()
    awaiting_feedback = False

def reset_interview():
    global awaiting_job_title, awaiting_answers, questions, answers, current_question
    
    awaiting_job_title = True
    awaiting_answers = False
    questions = []
    answers = []
    current_question = 0
    
    speak('Interview session ended. You can start a new session by stating the job title.')

def collect_answer(audio):
    global current_question, answers
    
    try:
        answer_audio_path = 'answer.wav'
        with open(answer_audio_path, 'wb') as f:
            f.write(audio.get_wav_data())
        
        answer = wav_to_text(answer_audio_path)
        if not answer:
            speak('Empty answer. Please speak again.')
        else:
            answers.append(answer)
            current_question += 1
            if current_question < len(questions):
                ask_question()
            else:
                provide_feedback()
                
    except Exception as e:
        print('Answer Error:', e)

# Define callback function for background listening
def callback(recognizer, audio):
    try:
        text = recognizer.recognize_google(audio)
        print(f"Recognized Text: {text}")
        if awaiting_job_title:
            get_job_title(audio)
        elif awaiting_answers:
            collect_answer(audio)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio.")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

def start_listening():
    recognizer = sr.Recognizer()

    # Initialize the microphone and start listening
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=2)  # Adjust for ambient noise
        print('You can start speaking now.')

        # Start listening in the background
        stop_listening = recognizer.listen_in_background(source, callback)

        # Keep the program running
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            stop_listening(wait_for_stop=False)
            print("Stopped listening.")

# Streamlit UI
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["About the platform", "Practice Mock", "Generate Content"])

# About Us Page
if page == "About the platform":
    st.title("About the AI Mock Interview Platform")
    st.write("""
        Welcome to the AI Mock Interview Platform. This platform is designed to help you prepare for your interviews by providing realistic mock interview scenarios.
        
        You can practice interviews for various domains such as Software Development, Data Science, Product Management, and more.
        
        Get started by selecting the 'Practice Mock' option in the sidebar.
    """)

# Practice Mock Interview Page
elif page == "Practice Mock":
    st.title("Practice Mock Interview")
    
    with st.form(key='mock_interview_form'):
        name = st.text_input("Enter your name:")
        domain = st.selectbox("Select your interview domain:", ["Software Developer", "Data Scientist", "Product Manager", "Other"])
        additional_info = st.text_area("Additional information about the interview (optional):")
        submit_button = st.form_submit_button(label='Start Interview')
    
    if submit_button:
        if awaiting_feedback:
            st.write("Please wait for the feedback of your previous interview before starting a new one.")
        else:
            st.write(f"Hello {name}, you have selected the {domain} domain for your interview.")
            st.write("Additional information:", additional_info)
            st.write("The interview will start shortly...")
            start_listening()
    
    if 'questions' in st.session_state:
        st.write("Interview Questions:")
        for i, question in enumerate(st.session_state['questions']):
            st.write(f"Q{i + 1}: {question}")

# Generate Content Page
elif page == "Generate Content":
    st.title("Generate Content for Interview Preparation")
    
    with st.form(key='generate_content_form'):
        content_title = st.text_input("Enter the title of the content:")
        generate_button = st.form_submit_button(label='Generate Content')
    
    if generate_button:
        convo.send_message(f"Generate interview preparation content for the topic: {content_title}")
        generated_content = convo.last.text
        
        st.write("Generated Content:")
        st.write(generated_content)
        
        convo.send_message(f"Generate 5 interview questions and answers for the topic: {content_title}")
        qna_content = convo.last.text
        
        st.write("Generated Interview Questions and Answers:")
        st.write(qna_content)
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, generated_content + "\n\n" + qna_content)
        pdf.output(f"{content_title}_Interview_Preparation.pdf")
        st.success("PDF generated successfully!")
