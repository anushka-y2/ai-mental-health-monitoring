# 💬 AI-Based Mental Health Monitoring Solution

An academic project that screens text for signs of mental health distress using NLP and machine learning, delivered through an interactive chatbot with mood tracking.

🔗 **Live App:** [ai-mental-health-monitor.streamlit.app](https://ai-mental-health-monitor.streamlit.app)

> ⚠️ **Disclaimer:** This is a screening aid built for an academic project. It is **not** a diagnostic tool and does not replace professional mental health care. If you or someone you know is in crisis, please contact a helpline (see below) or local emergency services.

---

## 📋 Overview

This project classifies user text into one of seven mental health categories (Normal, Depression, Anxiety, Stress, Bipolar, Suicidal, Personality Disorder) using a machine learning model trained on ~53,000 labeled real-world text samples. It combines this with a rule-based severity scale and safety-net keyword detection to provide more reliable, human-friendly responses through a chat interface — plus daily mood tracking and trend visualization.

## ✨ Features

- **Text classification** — TF-IDF + Logistic Regression model trained on labeled mental health text data
- **5-level severity scale** — Happy → Okay → Sad → Distressed → Crisis, blending ML predictions with keyword detection for more reliable results on short/ambiguous input
- **Safety-net layer** — high-risk phrases are always flagged as Crisis regardless of model confidence, with crisis helpline numbers shown immediately
- **Interactive chatbot GUI** — built with Streamlit, includes quick mood check-in buttons and a full chat interface
- **Persistent mood tracking** — every check-in is logged to a local SQLite database, viewable as a personal history and daily mood trend chart
- **Deployed live** — accessible via web browser, no installation required

## 🛠️ Tech Stack

| Component | Tools Used |
|---|---|
| Language | Python |
| NLP preprocessing | NLTK (tokenization, stemming, stopword removal) |
| Machine learning | scikit-learn (TF-IDF, Logistic Regression) |
| Data storage | SQLite |
| GUI / Deployment | Streamlit, Streamlit Community Cloud |
| Dataset | [Sentiment Analysis for Mental Health (Kaggle)](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health) |

## 📁 Project Structure

```
├── app.py              # Streamlit GUI application (main entry point)
├── train.py            # Model training script (run once to generate model files)
├── predict.py           # Standalone prediction/testing script
├── chatbot.py            # Terminal-based chatbot (early prototype)
├── mh_model.pkl          # Trained Logistic Regression model
├── vectorizer.pkl        # Fitted TF-IDF vectorizer
├── requirements.txt      # Python dependencies
└── mood_log.db           # SQLite database (auto-created on first run)
```

## 🚀 Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (only needed once, or if you want to retrain)
python train.py

# Launch the app
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

## 📊 Model Performance

Trained on 52,681 cleaned text samples (80/20 train-test split), achieving **76% overall accuracy** across 7 classes. Class imbalance was handled using `class_weight='balanced'`, and a custom stopword list preserves negation/intensity words (e.g., "not," "never," "all") that are otherwise stripped by default NLP pipelines but carry critical meaning in this domain.

## ⚖️ Ethical Considerations & Limitations

- This is a **screening tool, not a diagnostic instrument** — it identifies text patterns statistically associated with certain mental health categories in training data; it does not understand context, sarcasm, or individual circumstances the way a professional would.
- The dataset originates from self-reported social media/forum posts, which may not generalize to all populations or presentation styles.
- Data storage is a simple local SQLite file with no authentication — suitable for a demo, not for real-world sensitive data handling.
- Free-tier hosting means stored data is not guaranteed to persist indefinitely.
- **IBM Watson Assistant integration** (originally planned per course syllabus) was substituted with a custom rule-based + ML chatbot, since IBM Cloud's personal account signup requires a credit card, which was not accessible for this project. This is documented as a real-world accessibility constraint.

## 📞 Crisis Resources (India)

- **AASRA:** 91-9820466726
- **Kiran (Government, 24/7 toll-free):** 1800-599-0019

## 📚 Academic Context

Built as part of the Artificial Intelligence course project list, covering:
- **Module I:** Python programming, file handling
- **Module II:** Machine learning with scikit-learn (classification, evaluation metrics)
- **Module III:** NLP with NLTK (preprocessing, text classification)
- **Module V:** Cloud AI service integration concepts (substituted per above)

## 📄 License

Academic project — for educational purposes.
