import speech_recognition as sr
import google.generativeai as genai
import pyaudio
import os
import time
import warnings

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

def speak(text):
    player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    stream_start = False

    # Replace this block with your preferred TTS API call
    with open("response.wav", "wb") as f:
        f.write(genai.text_to_speech(text).content)
    # Play the audio file
    with open("response.wav", "rb") as f:
        data = f.read()
        player_stream.write(data)

def wav_to_text(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    text = ''.join(segment.text for segment in segments)
    return text

def prompt_gpt(audio):
    try:
        prompt_audio_path = 'prompt.wav'
        
        with open(prompt_audio_path, 'wb') as f:
            f.write(audio.get_wav_data())
        
        prompt_text = wav_to_text(prompt_audio_path)
        
        if len(prompt_text.strip()) == 0:
            print('Empty prompt. Please speak again.')
        else:
            print('User: ' + prompt_text)
            
            convo.send_message(prompt_text)
            output = convo.last.text
            
            print('Gemini: ', output)
            speak(output)
            
    except Exception as e:
        print('Prompt Error: ', e)

def callback(recognizer, audio):
    prompt_gpt(audio)

def start_listening():
    with source as s:
        r.adjust_for_ambient_noise(s, duration=2)
        
    print('You can start speaking now.')
    
    r.listen_in_background(source, callback)
    
    while True:
        time.sleep(0.5)

if __name__ == '__main__':
    start_listening()
