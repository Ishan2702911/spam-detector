"""
app.py
------
Streamlit app for Spam Email Classifier.
Run: streamlit run app.py
"""

import pickle
import re
import string
import streamlit as st
import nltk

# MUST be first streamlit command
st.set_page_config(
    page_title="SpamShield",
    page_icon="🛡️",
    layout="centered"
)

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

STOP_WORDS = set(stopwords.words("english"))
stemmer    = PorterStemmer()

# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vec = pickle.load(f)
    return model, vec

model, vectorizer = load_model()

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)

def predict(text: str):
    clean = preprocess(text)
    vec   = vectorizer.transform([clean])
    pred  = model.predict(vec)[0]
    try:
        prob = model.predict_proba(vec)[0][1]
    except AttributeError:
        # LinearSVC has no predict_proba — use decision function instead
        score = model.decision_function(vec)[0]
        # sigmoid approximation
        import math
        prob = 1 / (1 + math.exp(-score))
    return pred, prob

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d0f11;
    color: #d4dae0;
}
.main { background-color: #0d0f11; }

.stTextArea textarea {
    background-color: #141618 !important;
    color: #d4dae0 !important;
    border: 1px solid #2e353c !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.88rem !important;
}

.stButton > button {
    background-color: #00c896 !important;
    color: #0d0f11 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.6rem 2rem !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.05em !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

hr { border-color: #252a2f !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<p style="font-family:IBM Plex Mono,monospace;font-size:1.4rem;font-weight:600;color:#fff;margin-bottom:0;">'
    'SPAM<span style="color:#00c896;">SHIELD</span></p>',
    unsafe_allow_html=True
)
st.caption("Paste any email or SMS text below to instantly detect spam.")
st.divider()

# ── Sample buttons ────────────────────────────────────────────────────────────
SPAM_SAMPLE = (
    "WINNER! You have been selected as a lucky winner of $1,000 Walmart gift card. "
    "Call now to claim your FREE prize. Limited time offer! Reply WIN to 80488."
)
HAM_SAMPLE = (
    "Hey, are we still on for lunch tomorrow at 1pm? Let me know if the timing works for you."
)

col1, col2 = st.columns(2)
with col1:
    spam_click = st.button("Load Spam Example")
with col2:
    ham_click  = st.button("Load Ham Example")

# ── Input ─────────────────────────────────────────────────────────────────────
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

if spam_click:
    st.session_state.input_text = SPAM_SAMPLE
elif ham_click:
    st.session_state.input_text = HAM_SAMPLE

user_input = st.text_area(
    "Message",
    value=st.session_state.input_text,
    height=160,
    placeholder="Paste your email or message here...",
    label_visibility="collapsed"
)

predict_btn = st.button("🔍  Analyse Message")

# ── Result ────────────────────────────────────────────────────────────────────
if predict_btn:
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        pred, prob = predict(user_input)
        spam_pct = round(prob * 100, 1)
        ham_pct  = round((1 - prob) * 100, 1)

        st.divider()

        if pred == 1:
            st.error(f"### ⚠ SPAM DETECTED")
            st.metric(label="Spam Probability", value=f"{spam_pct}%")
            st.progress(prob)
            st.caption("This message shows strong spam signals — promotional language, urgency cues, or suspicious links detected.")
        else:
            st.success(f"### ✔ NOT SPAM")
            st.metric(label="Legitimate Probability", value=f"{ham_pct}%")
            st.progress(1 - prob)
            st.caption("This message appears to be legitimate. No strong spam indicators found.")

        # Show preprocessed tokens for transparency
        with st.expander("See how the model read this message"):
            clean = preprocess(user_input)
            st.code(clean, language=None)

# ── How it works ──────────────────────────────────────────────────────────────
st.divider()
with st.expander("How it works"):
    st.markdown("""
    1. **Preprocessing** — text is lowercased, numbers and punctuation removed, stop-words filtered, stemmed.
    2. **TF-IDF Vectorization** — converts cleaned text into a 5,000-feature vector using unigrams and bigrams.
    3. **Model** — trained on 5,574 labeled SMS messages (UCI Spam Collection). Naive Bayes, Logistic Regression, and SVM were benchmarked; best F1-score model was selected.
    4. **Output** — spam probability with a clear verdict.
    """)

st.markdown(
    '<p style="text-align:center;color:#5a6473;font-size:0.78rem;font-family:IBM Plex Mono,monospace;margin-top:2rem;">'
    'SpamShield · UCI SMS Spam Collection · Built with Streamlit</p>',
    unsafe_allow_html=True
)
