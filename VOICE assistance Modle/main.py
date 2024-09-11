import os
import requests
from gtts import gTTS
import speech_recognition as sr
import subprocess

# Set the API key for Gemini
os.environ['GEMINI_API_KEY'] = "AIzaSyCBBU-Z5VTrwTU9bZCcZbilVPLciyOEhMw"

# Function to preprocess user input
def preprocess_input(input_text):
    return input_text

# Function to respond to user input using Gemini API
def respond_to_input(input_text):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('GEMINI_API_KEY')}"
    }
    data = {
        "query": input_text
    }
    response = requests.post("https://api.gemini.ai/v1/text", headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['response']['text']
    else:
        return "Sorry, I couldn't process your request at the moment."

# Function to convert text to speech using gTTS
def speak_text(text):
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    subprocess.run(["afplay", "response.mp3"])  # macOS specific, use "aplay" for Linux, "start response.mp3" for Windows

# Function to capture speech input using SpeechRecognition
def capture_speech():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone(device_index=0)  # Adjust device index as needed

    with microphone as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        user_input = recognizer.recognize_google(audio)
        return user_input
    except sr.UnknownValueError:
        return "Sorry, I didn't catch that."
    except sr.RequestError as e:
        return f"Could not request results; {e}"

# Main function to interact with the user
def main():
    print("Hello! I'm a voice-enabled chatbot powered by Gemini.")
    print("You can speak to me or type 'exit' to quit.")
    while True:
        print("You: ", end="")
        user_input = capture_speech()
        print(user_input)
        if user_input.lower() == 'exit':
            print("Goodbye!")
            speak_text("Goodbye!")
            break
        else:
            response = respond_to_input(preprocess_input(user_input))
            print("Bot:", response)
            speak_text(response)

if __name__ == "__main__":
    main()
