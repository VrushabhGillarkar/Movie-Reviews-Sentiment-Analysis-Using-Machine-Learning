# Save as: train_model.py

import numpy as np
import pandas as pd
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 👉 Add this line to tell NLTK where to find the downloaded data
nltk.data.path.append("./nltk_data")

# 🚫 REMOVE these lines (Render doesn't allow downloads at runtime)
# nltk.download('punkt')
# nltk.download('stopwords')

# Load dataset
dataset = pd.read_csv('Dataset/IMDB.csv')
dataset.sentiment.replace({'positive': 1, 'negative': 0}, inplace=True)

# Cleaning functions
def clean_html(text):
    return re.sub(r'<.*?>', '', text)

def remove_special_chars(text):
    return ''.join([ch if ch.isalnum() else ' ' for ch in text])

def to_lower(text):
    return text.lower()

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    return [w for w in tokens if w not in stop_words]

def stem_text(words):
    stemmer = SnowballStemmer('english')
    return ' '.join([stemmer.stem(w) for w in words])

def preprocess(text):
    text = clean_html(text)
    text = remove_special_chars(text)
    text = to_lower(text)
    words = remove_stopwords(text)
    return stem_text(words)

# Preprocessing
dataset.review = dataset.review.apply(preprocess)

# Feature extraction
cv = CountVectorizer(max_features=2000)
X = cv.fit_transform(dataset.review).toarray()
y = dataset.sentiment.values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(model, 'model/MRSA_mnb.pkl')
joblib.dump(cv, 'model/MRSA_vectorizer.pkl')

# Evaluate
y_pred = model.predict(X_test)
print(f"\n🎯 Accuracy: {round(accuracy_score(y_test, y_pred)*100, 2)}%\n")
