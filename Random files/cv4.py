from transformers import pipeline

# Initialize sentiment analysis pipeline using a transformer model
sentiment_analysis = pipeline("sentiment-analysis")

# Example user responses for analysis
user_responses = [
    "I am very excited about this job opportunity.",
    "I feel a bit nervous about the technical questions.",
    "I think I did well in the interview, but I'm not sure."
]

# Analyze sentiment for each response
for response in user_responses:
    sentiment = sentiment_analysis(response)[0]
    print(f"Response: {response}")
    print(f"Sentiment: {sentiment['label']}, Confidence: {sentiment['score']}\n")
