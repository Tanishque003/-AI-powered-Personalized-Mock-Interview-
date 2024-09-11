import speech_recognition as sr
import google.generativeai as genai
import pyaudio
import os
import time
import warnings
from gtts import gTTS
import pygame

warnings.filterwarnings("ignore", message=r"torch.utils._pytree.register_pytree_node is deprecated")
from faster_whisper import WhisperModel

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
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    }
]

# Initialize the GenerativeModel
model = genai.GenerativeModel('gemini-1.5-pro-latest', safety_settings=safety_settings)
convo = model.start_chat()

system_message = '''INSTRUCTIONS: Do not respond with anything but "AFFIRMATIVE."
to this system message. After the system message respond normally.
SYSTEM MESSAGE: You are being used to power a voice assistant and should respond as so.
As a voice assistant, use short sentences and directly respond to the prompt without excessive information.
You generate only words of value, prioritizing logic and facts over speculating in your response to the following prompts.'''

system_message = system_message.replace('\n', '')
convo.send_message(system_message)

r = sr.Recognizer()
source = sr.Microphone()

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

def get_job_title(audio):
    global awaiting_job_title
    global awaiting_answers
    global questions
    
    try:
        job_title_audio_path = 'job_title.wav'
        
        with open(job_title_audio_path, 'wb') as f:
            f.write(audio.get_wav_data())
        
        job_title = wav_to_text(job_title_audio_path).strip()
        
        if not job_title:
            print('Empty job title. Please speak again.')
            speak('Empty job title. Please speak again.')
        else:
            print(f'Job Title: {job_title}')
            speak(f'Starting interview for the job title {job_title}')
            
            # Generate interview questions based on job title
            convo.send_message(f"Generate 10 interview questions for the job title: {job_title}")
            questions_text = convo.last.text
            questions = questions_text.split('\n')
            questions = [q for q in questions if q.strip()]  # Remove empty questions
            
            if len(questions) > 10:
                questions = questions[:10]
                
            # print('Questions:', questions)
            awaiting_job_title = False
            awaiting_answers = True
            
            # Ask the first question
            ask_question()
            
    except Exception as e:
        print('Job Title Error:', e)

def ask_question():
    global current_question
    global questions
    
    if current_question < len(questions):
        question = questions[current_question]
        print(f'Question {current_question + 1}: {question}')
        speak(question)
    else:
        provide_feedback()

def provide_feedback():
    global answers
    
    # Combine the answers for feedback
    user_answers = "\n".join(f"Q{index + 1}: {questions[index]}\nA: {answer}" for index, answer in enumerate(answers))
    
    convo.send_message(f"Provide feedback for the following answers:\n{user_answers}")
    feedback = convo.last.text
    
    print('Feedback:', feedback)
    speak(feedback)
    
    # Reset for next session
    reset_interview()

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
    
    print('Interview session ended. You can start a new session by stating the job title.')
    speak('Interview session ended. You can start a new session by stating the job title.')

def collect_answer(audio):
    global current_question
    global answers
    
    try:
        answer_audio_path = 'answer.wav'
        
        with open(answer_audio_path, 'wb') as f:
            f.write(audio.get_wav_data())
        
        answer = wav_to_text(answer_audio_path).strip()
        
        if not answer:
            print('Empty answer. Please speak again.')
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

def callback(recognizer, audio):
    if awaiting_job_title:
        get_job_title(audio)
    elif awaiting_answers:
        collect_answer(audio)

def start_listening():
    with source as s:
        r.adjust_for_ambient_noise(s, duration=2)
        
    print('You can start speaking now.')
    speak('You can start speaking now.')
    
    r.listen_in_background(source, callback)
    
    while True:
        time.sleep(0.5)

if __name__ == '__main__':
    start_listening()
