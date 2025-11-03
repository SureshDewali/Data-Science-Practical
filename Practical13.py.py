import nltk
from nltk.stem import WordNetLemmatizer
import random

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

intents = {
    "greeting": {
        "patterns": ["hello", "hi", "hey", "good morning", "good evening"],
        "responses": ["Hello! How can I help you?", "Hi there! 😊", "Hey! What’s up?"]
    },
    "goodbye": {
        "patterns": ["bye", "see you", "goodbye"],
        "responses": ["Goodbye! Have a great day! 👋", "See you later!", "Bye-bye!"]
    },
    "feeling": {
        "patterns": ["how are you", "how are u", "how’s it going"],
        "responses": ["I’m doing great! How about you?", "Feeling awesome 😄", "All good! What about you?"]
    },
    "name": {
        "patterns": ["what is your name", "who are you", "tell me your name"],
        "responses": ["I’m your Data Science Assistant 🤖", "You can call me DS Bot!", "I’m Suresh’s friendly chatbot!"]
    },
    "creator": {
        "patterns": ["who created you", "who made you", "who built you"],
        "responses": ["I was created by Suresh Dewali for a Data Science project! 🚀", "Suresh is my creator 😄", "I was built as part of a machine learning practical!"]
    },
    "help": {
        "patterns": ["help me", "i need help", "can you help me"],
        "responses": ["Sure! What do you need help with?", "Of course! Tell me your question.", "I’m here to help you 🙂"]
    },
    "data_science": {
        "patterns": ["what is data science", "tell me about data science", "explain data science"],
        "responses": [
            "Data Science is the study of data using statistics, programming, and machine learning.",
            "It combines maths, coding, and AI to extract knowledge from data 📊.",
            "Data Science helps us make smart decisions from raw data."
        ]
    },
    "thanks": {
        "patterns": ["thank you", "thanks", "thanks a lot"],
        "responses": ["You’re welcome! 😊", "Glad to help!", "Anytime, Suresh!"]
    },
    "default": {
        "responses": [
            "I didn’t understand that. Could you say it differently?",
            "Hmm... I’m still learning that.",
            "Can you rephrase your question?"
        ]
    }
}

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def match_intent(user_input):
    tokens = preprocess(user_input)
    best_intent = None
    best_score = 0

    for intent, data in intents.items():
        # Skip the default intent during pattern matching
        if "patterns" not in data:
            continue

        for pattern in data["patterns"]:
            pattern_tokens = preprocess(pattern)
            common_words = len(set(tokens) & set(pattern_tokens))
            score = common_words / len(pattern_tokens)
            if score > best_score:
                best_score = score
                best_intent = intent

    if best_score > 0.4 and best_intent:
        return random.choice(intents[best_intent]["responses"])
    else:
        return random.choice(intents["default"]["responses"])

print("Name: Suresh Dewali")
print("Roll No: 1323575\n")
print("🤖 Chatbot: Hello! I’m your Data Science Assistant. Type 'bye' to exit.\n")

while True:
    user_input = input("You: ").lower()
    if user_input in ["bye", "exit", "quit"]:
        print("🤖 Chatbot: Goodbye! Have a great day! 👋")
        break
    response = match_intent(user_input)
    print("🤖 Chatbot:", response)
