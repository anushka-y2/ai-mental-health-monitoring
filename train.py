print("SCRIPT STARTED")

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

df = pd.read_csv('Combined Data.csv')
df = df.drop(columns=['Unnamed: 0'])

print(df.isnull().sum())
df = df.dropna(subset=['statement'])
print(df.shape)

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

stemmer = PorterStemmer()

default_stopwords = set(stopwords.words('english'))
keep_words = {'not', 'no', 'never', 'all', 'nothing', 'none', 'only', 'cant', 'cannot'}
stop_words = default_stopwords - keep_words

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 1]
    return ' '.join(tokens)

print("Cleaning text... this takes a few minutes on 52k rows.")
df['clean_text'] = df['statement'].apply(clean_text)
df.to_csv('cleaned_data.csv', index=False)
print("Cleaning done.")
print(df[['statement', 'clean_text']].head())

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['status'],
    test_size=0.2, random_state=42, stratify=df['status']
)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, 'mh_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("Model and vectorizer saved: mh_model.pkl, vectorizer.pkl")