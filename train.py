import pandas as pd
import pickle
import urllib.request
import zipfile
import os
import re
import string
import nltk

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

STOP_WORDS = set(stopwords.words("english"))
stemmer    = PorterStemmer()

# 1. Download dataset
URL      = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
ZIP_FILE = "smsspamcollection.zip"
TXT_FILE = "SMSSpamCollection"

if not os.path.exists(TXT_FILE):
    print("Downloading dataset...")
    urllib.request.urlretrieve(URL, ZIP_FILE)
    with zipfile.ZipFile(ZIP_FILE, "r") as z:
        z.extractall(".")
    os.remove(ZIP_FILE)

# 2. Load
df = pd.read_csv(TXT_FILE, sep="\t", header=None, names=["label", "text"])
print(f"Total: {len(df)} | Spam: {(df.label=='spam').sum()} | Ham: {(df.label=='ham').sum()}")
df["label_enc"] = df["label"].map({"ham": 0, "spam": 1})

# 3. Preprocess
def preprocess(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)

print("Preprocessing...")
df["clean"] = df["text"].apply(preprocess)

# 4. Vectorize — char + word level helps catch spam patterns
vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    sublinear_tf=True
)
X = vectorizer.fit_transform(df["clean"])
y = df["label_enc"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Compare models
models = {
    "Naive Bayes"        : MultinomialNB(alpha=0.2),
    "Logistic Regression": LogisticRegression(max_iter=1000, C=5, random_state=42),
    "SVM"                : LinearSVC(C=1.0, random_state=42, max_iter=2000),
}

print(f"\n{'Model':<25} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
print("-" * 60)

best_model, best_f1, best_name = None, 0, ""

for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    print(f"{name:<25} {acc*100:>6.2f}% {prec*100:>6.2f}% {rec*100:>6.2f}% {f1*100:>6.2f}%")
    if f1 > best_f1:
        best_f1, best_model, best_name = f1, clf, name

print(f"\nBest: {best_name} (F1: {best_f1*100:.2f}%)")
print(classification_report(y_test, best_model.predict(X_test), target_names=["Ham","Spam"]))

# 6. Save
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Saved model.pkl and vectorizer.pkl")
