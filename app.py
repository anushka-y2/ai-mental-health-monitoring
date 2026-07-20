import streamlit as st
import joblib
import re
import nltk
import csv
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from datetime import datetime, date
import random
import pandas as pd

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

stemmer = PorterStemmer()
default_stopwords = set(stopwords.words('english'))
keep_words = {'not', 'no', 'never', 'all', 'nothing', 'none', 'only', 'cant', 'cannot'}
stop_words = default_stopwords - keep_words

LOG_FILE = "mood_log.csv"

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model = joblib.load('mh_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 1]
    return ' '.join(tokens)

# ================= SEVERITY SCALE =================
SEVERITY_LEVELS = {
    0: {"label": "Happy",      "emoji": "😄", "color": "#2ecc71"},
    1: {"label": "Okay",       "emoji": "🙂", "color": "#3498db"},
    2: {"label": "Sad",        "emoji": "😔", "color": "#f39c12"},
    3: {"label": "Distressed", "emoji": "😟", "color": "#e67e22"},
    4: {"label": "Crisis",     "emoji": "🚨", "color": "#e74c3c"},
}

KEYWORD_LEVELS = {
    0: ['happy', 'great', 'good day', 'excited', 'joyful', 'wonderful', 'grateful',
        'amazing', 'awesome', 'content', 'good', 'nice', 'fantastic', 'delighted'],
    1: ['okay', 'fine', 'alright', 'not bad', 'meh', 'so so', 'decent', 'neutral'],
    2: ['sad', 'down', 'unhappy', 'low', 'upset', 'lonely', 'tired of everything',
        'crying', 'bad', 'not good', 'not great', 'rough day', 'terrible day'],
    3: ['depressed', 'anxious', 'overwhelmed', "can't cope", 'panic', 'worthless',
        'hopeless', 'exhausted', 'breaking down', 'awful', 'horrible'],
    4: ['end it all', 'ending it all', 'kill myself', 'want to die', 'not worth living',
        'better off dead', 'no reason to live', 'suicidal', 'suicide', 'hurt myself', 'self harm'],
}

CATEGORY_TO_SEVERITY = {
    'Normal': 1, 'Stress': 2, 'Anxiety': 2, 'Bipolar': 2,
    'Personality disorder': 2, 'Depression': 3, 'Suicidal': 4,
}

def keyword_severity(raw_text):
    text_lower = raw_text.lower()
    for level in sorted(KEYWORD_LEVELS.keys(), reverse=True):
        for phrase in KEYWORD_LEVELS[level]:
            if phrase in text_lower:
                return level
    return None

def predict_status(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    confidence = max(proba)

    kw_level = keyword_severity(text)
    ml_level = CATEGORY_TO_SEVERITY.get(pred, 1)

    if kw_level is not None:
        if kw_level == 4 or ml_level == 4:
            final_level = 4
        else:
            final_level = kw_level
    else:
        final_level = ml_level

    if final_level == 4:
        pred = 'Suicidal'
        confidence = 1.0

    return pred, confidence, final_level

# ================= RESPONSES =================
RESPONSES = {
    0: ["That's wonderful to hear! What's been going well for you?",
        "Love that energy! What made today good?"],
    1: ["Okay is a fine place to be. Anything on your mind?",
        "Glad you're holding steady. How's everything else going?"],
    2: ["I'm sorry you're feeling sad. Want to tell me more about what's going on?",
        "That sounds tough. What's been weighing on you?"],
    3: ["That sounds really overwhelming. Have you been able to talk to someone you trust?",
        "I hear how hard this is. What's feeling most heavy right now?"],
    4: ["I'm really concerned about what you're sharing. Please reach out to a crisis helpline right now.\n\n"
        "📞 **AASRA: 91-9820466726**\n📞 **Kiran: 1800-599-0019** (24/7, toll-free)"],
}

CLOSING_LINES = {
    0: ["Keep that momentum going! 🌟", "Here's to more days like this one."],
    1: ["Steady days matter too — take care of yourself.", "Hope the rest of your day goes smoothly."],
    2: ["This feeling won't last forever, even when it feels like it will. Be gentle with yourself today.",
        "It's okay to not be okay. Small steps count — you don't have to fix everything today."],
    3: ["You're carrying a lot right now, and that takes real strength. Please consider reaching out to someone you trust.",
        "It's okay to ask for help — you don't have to handle this alone."],
    4: ["You matter, and support is available right now. Please reach out to one of the numbers above — someone wants to help."],
}

def get_response(severity_level):
    return random.choice(RESPONSES[severity_level])

def get_closing_line(severity_level):
    return random.choice(CLOSING_LINES[severity_level])

import sqlite3

DB_FILE = "mood_log.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS mood_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            entry_date TEXT NOT NULL,
            entry_time TEXT NOT NULL,
            message TEXT NOT NULL,
            category TEXT NOT NULL,
            severity INTEGER NOT NULL,
            confidence REAL NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def log_entry(username, message, category, severity, confidence):
    init_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO mood_entries (username, entry_date, entry_time, message, category, severity, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        username, date.today().isoformat(), datetime.now().strftime("%H:%M:%S"),
        message, category, severity, round(confidence, 2)
    ))
    conn.commit()
    conn.close()

def load_user_history(username):
    init_db()
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query(
        "SELECT * FROM mood_entries WHERE username = ? ORDER BY id",
        conn, params=(username,)
    )
    conn.close()
    return df

# ================= STREAMLIT UI =================
st.set_page_config(page_title="Mental Health Companion", page_icon="💬", layout="centered")

if "username" not in st.session_state:
    st.session_state.username = None

if st.session_state.username is None:
    st.title("💬 Mental Health Screening Companion")
    st.write("Enter a name or nickname to start your private check-in log.")
    name_input = st.text_input("Your name / nickname")
    if st.button("Start") and name_input.strip():
        st.session_state.username = name_input.strip()
        st.rerun()
    st.stop()

username = st.session_state.username

if "history" not in st.session_state:
    st.session_state.history = []

# ---- Sidebar ----
with st.sidebar:
    st.header(f"👋 Hi, {username}")
    st.header("🧭 Quick Check-in")
    quick_options = {
        "😄 Happy": "I'm feeling really happy today",
        "🙂 Okay": "I'm feeling okay, nothing special",
        "😔 Sad": "I'm feeling sad today",
        "😟 Overwhelmed": "I'm feeling really overwhelmed and anxious",
    }
    quick_choice = None
    for label, phrase in quick_options.items():
        if st.button(label, use_container_width=True):
            quick_choice = phrase

    st.divider()
    st.header("📞 Crisis Resources")
    st.markdown("**AASRA:** 91-9820466726")
    st.markdown("**Kiran (Govt, 24/7):** 1800-599-0019")
    st.caption("If you or someone you know is in immediate danger, contact local emergency services.")

    st.divider()
    if st.button("🗑️ Clear this session", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    if st.button("🔒 Switch user", use_container_width=True):
        st.session_state.username = None
        st.session_state.history = []
        st.rerun()

st.title("💬 Mental Health Screening Companion")
st.caption("An academic screening demo — not a diagnostic tool or substitute for professional care.")

tab1, tab2 = st.tabs(["💬 Chat", "📊 My History & Trends"])

with tab1:
    user_input = st.chat_input("How are you feeling today?")
    final_input = quick_choice or user_input

    if final_input:
        status, confidence, severity = predict_status(final_input)
        reply = get_response(severity)
        closing = get_closing_line(severity)
        st.session_state.history.append({
            "user": final_input, "status": status, "confidence": confidence,
            "severity": severity, "reply": reply, "closing": closing,
            "time": datetime.now().strftime("%H:%M")
        })
        log_entry(username, final_input, status, severity, confidence)

        # Gentle nudge after 3+ consecutive low-mood messages
        recent = [h["severity"] for h in st.session_state.history[-3:]]
        if len(recent) == 3 and all(s >= 2 for s in recent):
            st.session_state.history.append({
                "user": None, "status": None, "confidence": None,
                "severity": None, "reply": None,
                "closing": "I've noticed you've been having a rough few check-ins. "
                           "It might help to talk to someone you trust, or a counselor — you don't have to carry this alone. 💙",
                "time": datetime.now().strftime("%H:%M")
            })

    for entry in st.session_state.history:
        with st.chat_message("user" if entry["user"] else "assistant"):
            if entry["user"]:
                st.write(entry["user"])
            else:
                st.info(entry["closing"])
                continue
        if entry["status"] is not None:
            level_info = SEVERITY_LEVELS[entry["severity"]]
            with st.chat_message("assistant"):
                st.write(entry["reply"])
                st.markdown(f"*{entry['closing']}*")
                st.markdown(
                    f"<span style='background-color:{level_info['color']}; color:white; padding:3px 10px; "
                    f"border-radius:10px; font-size:0.8em;'>{level_info['emoji']} {level_info['label']}</span> "
                    f"<span style='font-size:0.75em; color:gray;'>· {entry['time']} · {entry['status']} "
                    f"({entry['confidence']:.2f})</span>", unsafe_allow_html=True
                )
                if entry["severity"] == 4:
                    st.error("⚠️ High-risk message detected. Please reach out to a crisis helpline.")

with tab2:
    st.subheader(f"📊 {username}'s Mood History")
    user_df = load_user_history(username)

    if user_df.empty:
        st.info("No history yet — start chatting to build your mood log!")
    else:
        st.line_chart(user_df.set_index(user_df.index)['severity'])
        st.caption("0 = Happy · 1 = Okay · 2 = Sad · 3 = Distressed · 4 = Crisis")

        st.markdown("### Daily average mood")
        daily_avg = user_df.groupby('entry_date')['severity'].mean()
        st.bar_chart(daily_avg)

        st.markdown("### Full log")
        st.dataframe(user_df[['entry_date', 'entry_time', 'message', 'category', 'severity']], use_container_width=True)

        csv_data = user_df.to_csv(index=False)
        st.download_button("⬇️ Download my data (CSV)", csv_data, file_name=f"{username}_mood_log.csv")

st.markdown("---")
st.caption("⚠️ This tool is a screening aid built for an academic project. It is not a diagnosis and does not replace professional mental health support.")