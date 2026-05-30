# Spam Email Classifier

NLP-based spam detector trained on 5,500+ SMS messages. Built with Streamlit.

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (run once)
```bash
python train.py
```
Downloads dataset automatically. Saves: `model.pkl`, `vectorizer.pkl`

### 3. Run the app
```bash
streamlit run app.py
```

## Deploy to Streamlit Cloud (Free)

1. Push this folder to a GitHub repo
2. Go to share.streamlit.io → New app
3. Select repo → set Main file path: `app.py`
4. Click Deploy

> Add `python train.py` as a pre-run step, or commit the `.pkl` files directly to the repo.

## Tech Stack
- Python, Scikit-learn, NLTK, Streamlit
- Multinomial Naive Bayes (best F1 among NB, LR, SVM)
- TF-IDF vectorization (5,000 features, unigrams + bigrams)
- Trained on UCI SMS Spam Collection (5,574 messages)
