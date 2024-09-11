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

wake_word = 'Google'
listening_for_wake_word = True

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

def listen_for_wake_word(audio):
    global listening_for_wake_word

    wake_audio_path = 'wake_detect.wav'
    with open(wake_audio_path, 'wb') as f:
        f.write(audio.get_wav_data())
    text_input = wav_to_text(wake_audio_path)

    if wake_word in text_input.lower().strip():
        print('Wake word detected. Please speak your prompt to JARVIS.')
        listening_for_wake_word = False

def prompt_gpt(audio):
    global listening_for_wake_word

    try:
        prompt_audio_path = 'prompt.wav'
        with open(prompt_audio_path, 'wb') as f:
            f.write(audio.get_wav_data())

        prompt_text = wav_to_text(prompt_audio_path)

        if len(prompt_text.strip()) == 0:
            print('Empty prompt. Please speak again.')
            listening_for_wake_word = True
        else:
            print('User: ' + prompt_text)
            convo.send_message(prompt_text)
            output = convo.last.text

            print('JARVIS: ', output)
            speak(output)

            print('\nSay', wake_word, 'to wake me up.\n')
            listening_for_wake_word = True

    except Exception as e:
        print('Prompt Error: ', e)

def callback(recognizer, audio):
    global listening_for_wake_word

    if listening_for_wake_word:
        listen_for_wake_word(audio)
    else:
        prompt_gpt(audio)

def start_listening():
    with source as s:
        r.adjust_for_ambient_noise(s, duration=2)

    print('\nSay', wake_word, 'to wake me up.\n')
    r.listen_in_background(source, callback)

    while True:
        time.sleep(0.5)

if __name__ == '__main__':
    start_listening()
