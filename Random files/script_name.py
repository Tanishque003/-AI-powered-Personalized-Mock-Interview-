import os
from transformers import pipeline
from gtts import gTTS
from faster_whisper import WhisperModel
import speech_recognition as sr

# Load pre-trained models
generator = pipeline("text-generation", model="distilgpt2")
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
whisper_model = WhisperModel("base", device="cpu")  # Use "cuda" for GPU
recognizer = sr.Recognizer()

# Function to generate interview questions
def generate_interview_question(job_title):
    prompt = f"Generate an interview question for a {job_title} role:"
    question = generator(prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
    return question.strip()

# Function to recognize speech using Whisper
def whisper_recognize(audio_path):
    segments, info = whisper_model.transcribe(audio_path)
    transcript = " ".join([segment.text for segment in segments])
    return transcript

# Function to recognize speech using Google Speech Recognition
def google_recognize():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print("You said: " + text)
            return text
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")

# Function to analyze sentiment
def analyze_sentiment(text):
    result = sentiment_analyzer(text)[0]
    return result

# Function to convert text to speech
def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")
    os.system("mpg321 output.mp3")  # Use an appropriate audio player for your system

# Function for real-time interaction
def real_time_interaction(job_title):
    question = generate_interview_question(job_title)
    print("System: " + question)
    
    # Use Google Speech Recognition for real-time response
    user_response = google_recognize()
    
    if user_response:
        sentiment = analyze_sentiment(user_response)
        print(f"Sentiment Analysis: {sentiment}")
        
        # Provide feedback based on sentiment
        feedback = f"Your response was {sentiment['label']} with a confidence score of {sentiment['score']:.2f}."
        print("System: " + feedback)
        text_to_speech(feedback)

# Example usage
if __name__ == "__main__":
    job_title = "Software Engineer"
    real_time_interaction(job_title)
