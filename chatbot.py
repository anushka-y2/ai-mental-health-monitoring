import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import random

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

stemmer = PorterStemmer()
default_stopwords = set(stopwords.words('english'))
keep_words = {'not', 'no', 'never', 'all', 'nothing', 'none', 'only', 'cant', 'cannot'}
stop_words = default_stopwords - keep_words

model = joblib.load('mh_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# ---- Text cleaning (same as training) ----
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 1]
    return ' '.join(tokens)

# ---- Safety-net keyword check ----
HIGH_RISK_PHRASES = [
    'end it all', 'ending it all', 'kill myself', 'want to die',
    'not worth living', 'better off dead', 'no reason to live',
    'suicidal', 'suicide', 'hurt myself', 'self harm'
]

def safety_check(raw_text):
    text_lower = raw_text.lower()
    return any(phrase in text_lower for phrase in HIGH_RISK_PHRASES)

# ---- Prediction ----
def predict_status(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    confidence = max(proba)

    if safety_check(text) and pred != 'Suicidal':
        pred = 'Suicidal'
        confidence = 1.0

    return pred, confidence

# ---- Rule-based chatbot responses ----
RESPONSES = {
    'Suicidal': [
        "I'm really concerned about what you're sharing. You don't have to go through this alone — please reach out to a crisis helpline right now. In India, you can call the AASRA helpline at 91-9820466726, or the Kiran helpline at 1800-599-0019 (24/7, toll-free).",
        "What you're feeling matters, and it's serious enough that I want you to talk to someone trained to help. Please contact a crisis line: Kiran (1800-599-0019) or AASRA (91-9820466726)."
    ],
    'Depression': [
        "It sounds like things have felt really heavy lately. Have you been able to talk to someone you trust about this?",
        "That sounds really hard to carry. Would it help to talk about what's been weighing on you most?"
    ],
    'Anxiety': [
        "That sounds overwhelming. Sometimes slow, deep breathing for a minute can help in the moment — would you like to try that?",
        "It makes sense that would feel stressful. What's been on your mind the most?"
    ],
    'Stress': [
        "Sounds like a lot is piling up right now. What feels like the biggest source of pressure?",
        "That's a lot to manage. Is there one thing on your plate you could set aside for now?"
    ],
    'Normal': [
        "Glad to hear you're doing okay! Anything on your mind you'd like to talk through anyway?",
        "That's good to hear. Is there anything you'd like to reflect on today?"
    ],
    'Bipolar': [
        "Thanks for sharing that. How have your energy levels and mood been shifting lately?",
        "I appreciate you telling me. Have these ups and downs been manageable lately?"
    ],
    'Personality disorder': [
        "I hear you. Would it help to talk through what's been on your mind?",
        "Thanks for opening up. What's been the hardest part of your day?"
    ]
}

DISCLAIMER = (
    "\n[Note: I'm a screening demo, not a licensed professional. "
    "For real support, please talk to a counselor, therapist, or trusted person.]\n"
)

def chatbot_response(predicted_status):
    return random.choice(RESPONSES.get(predicted_status, ["Tell me more about how you're feeling."]))

# ---- Main chat loop ----
def run_chat():
    print("=" * 60)
    print("Mental Health Screening Chatbot (Demo)")
    print("Type 'quit' to exit.")
    print(DISCLAIMER)
    print("=" * 60)

    while True:
        user_input = input("\nYou: ")
        if user_input.strip().lower() in ['quit', 'exit']:
            print("Bot: Take care. Reach out anytime.")
            break

        status, confidence = predict_status(user_input)
        reply = chatbot_response(status)

        print(f"Bot: {reply}")
        print(f"[Detected category: {status} | confidence: {confidence:.2f}]")

        if status == 'Suicidal':
            print(DISCLAIMER)

if __name__ == "__main__":
    run_chat()