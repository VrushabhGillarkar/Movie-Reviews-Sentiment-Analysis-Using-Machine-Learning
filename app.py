# app.py

from flask import Flask, render_template, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('Models/MRSA_mnb.pkl')
vectorizer = joblib.load('Models/MRSA_vectorizer.pkl')

# Text preprocessing functions
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

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Normal form submission route
@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    processed = preprocess(review)
    vectorized = vectorizer.transform([processed]).toarray()
    
    prediction = model.predict(vectorized)[0]
    proba = model.predict_proba(vectorized)[0]
    confidence = round(100 * max(proba), 2)

    sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"

    return render_template('result.html', review=review, sentiment=sentiment, confidence=confidence)

# ✅ New: Live AJAX prediction route
@app.route('/predict_live', methods=['POST'])
def predict_live():
    data = request.get_json()
    text = data.get('text', '')

    if not text.strip():
        return jsonify({'sentiment': 'neutral', 'confidence': 0.0})

    processed = preprocess(text)
    vectorized = vectorizer.transform([processed]).toarray()
    prediction = model.predict(vectorized)[0]
    proba = model.predict_proba(vectorized)[0]
    confidence = round(100 * max(proba), 2)

    sentiment = 'positive' if prediction == 1 else 'negative'

    return jsonify({'sentiment': sentiment, 'confidence': confidence})

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
