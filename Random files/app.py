from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import google.generativeai as genai
import os
import time
from gtts import gTTS
import pygame
from faster_whisper import WhisperModel

app = Flask(__name__)

# Initialize models and configurations
whisper_size = "base"
num_cores = os.cpu_count()
whisper_model = WhisperModel(
    whisper_size,
    device='cpu',
    compute_type='int8',
    cpu_threads=num_cores,
    num_workers=num_cores
)

GOOGLE_API_KEY = "AIzaSyCBBU-Z5VTrwTU9bZCcZbilVPLciyOEhMw"
genai.configure(api_key=GOOGLE_API_KEY)

generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
]

# Initialize the GenerativeModel
model = genai.GenerativeModel('gemini-1.5-pro-latest', safety_settings=safety_settings)
convo = model.start_chat()

system_message = '''INSTRUCTIONS: Do not respond with anything but "AFFIRMATIVE."
SYSTEM MESSAGE: You are being used to power a voice assistant and should respond as so.
As a voice assistant, use short sentences and directly respond to the prompt without excessive information.
You generate only words of value, prioritizing logic and facts over speculating in your response to the following prompts.'''

system_message = system_message.replace('\n', '')
convo.send_message(system_message)

# Global variables
awaiting_job_title = True
awaiting_answers = False
questions = []
answers = []
current_question = 0

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
    text = ''.join(segment.text for segment in segments)
    return text

def get_job_title(job_title_audio_path):
    global awaiting_job_title
    global awaiting_answers
    global questions

    job_title = wav_to_text(job_title_audio_path).strip()

    if not job_title:
        return 'Empty job title. Please speak again.'
    
    convo.send_message(f"Generate 10 interview questions for the job title: {job_title}")
    questions_text = convo.last.text
    questions = questions_text.split('\n')
    questions = [q for q in questions if q.strip()]

    if len(questions) > 10:
        questions = questions[:10]

    awaiting_job_title = False
    awaiting_answers = True
    return 'Starting interview for the job title ' + job_title

def ask_question():
    global current_question
    global questions

    if current_question < len(questions):
        question = questions[current_question]
        return question
    else:
        return provide_feedback()

def provide_feedback():
    global answers

    user_answers = "\n".join(f"Q{index + 1}: {questions[index]}\nA: {answer}" for index, answer in enumerate(answers))
    
    convo.send_message(f"Provide feedback for the following answers:\n{user_answers}")
    feedback = convo.last.text

    return feedback

def reset_interview():
    global awaiting_job_title
    global awaiting_answers
    global questions
    global answers
    global current_question

    awaiting_job_title = True
    awaiting_answers = False
    questions = []
    answers = []
    current_question = 0

    return 'Interview session ended. You can start a new session by stating the job title.'

def collect_answer(answer_audio_path):
    global current_question
    global answers

    answer = wav_to_text(answer_audio_path).strip()

    if not answer:
        return 'Empty answer. Please speak again.'

    answers.append(answer)
    current_question += 1

    if current_question < len(questions):
        return ask_question()
    else:
        return provide_feedback()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_interview', methods=['POST'])
def start_interview():
    job_title = request.form['job_title']
    resume = request.files['resume']

    # Save resume to file
    resume_path = 'resume.pdf'
    resume.save(resume_path)

    # Process resume if needed

    # Simulate interview
    return jsonify(message='Interview started for job title ' + job_title)

@app.route('/get_question', methods=['GET'])
def get_question():
    question = ask_question()
    return jsonify(question=question)

@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    answer_audio = request.files['answer']
    answer_audio_path = 'answer.wav'
    answer_audio.save(answer_audio_path)

    feedback = collect_answer(answer_audio_path)
    return jsonify(feedback=feedback)

if __name__ == '__main__':
    app.run(debug=True)
