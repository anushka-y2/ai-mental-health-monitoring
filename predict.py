import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

stemmer = PorterStemmer()
default_stopwords = set(stopwords.words('english'))
keep_words = {'not', 'no', 'never', 'all', 'nothing', 'none', 'only', 'cant', 'cannot'}
stop_words = default_stopwords - keep_words

model = joblib.load('mh_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 1]
    return ' '.join(tokens)

HIGH_RISK_PHRASES = [
    'end it all', 'ending it all', 'kill myself', 'want to die',
    'not worth living', 'better off dead', 'no reason to live',
    'suicidal', 'suicide'
]

def safety_check(raw_text):
    text_lower = raw_text.lower()
    return any(phrase in text_lower for phrase in HIGH_RISK_PHRASES)

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

if __name__ == "__main__":
    samples = [
        "I can't stop thinking about ending it all",
        "Feeling great today, went for a run and had a good breakfast",
        "My heart is racing and I can't calm down, everything feels overwhelming"
    ]
    for s in samples:
        label, conf = predict_status(s)
        print(f"Text: {s}\nPredicted: {label} (confidence: {conf:.2f})\n")