# 🎬 Movie Reviews Sentiment Analysis using Machine Learning

A **Flask web application** that classifies movie reviews as **positive** or **negative** using Natural Language Processing and a trained **Multinomial Naive Bayes** model. The app also displays a **confidence score** for each prediction.

---

## 🚀 Features

- 🔍 Real-time movie review sentiment prediction
- 🧠 ML model trained on preprocessed data
- 📊 Displays sentiment & prediction confidence
- 💡 Live AJAX endpoint for dynamic predictions
- 🎨 Clean and responsive frontend UI

---

## 🧰 Tech Stack

- **Backend**: Flask
- **ML/NLP**: Scikit-learn, NLTK
- **Frontend**: HTML, CSS, JavaScript (AJAX)
- **Model Serialization**: Joblib

---

## 📁 Project Structure

```bash
.
├── Models/
│   ├── MRSA_mnb.pkl              # Trained model
│   └── MRSA_vectorizer.pkl       # TF-IDF vectorizer
├── static/                       # Static files (CSS/JS)
├── templates/
│   ├── index.html                # Home page template
│   └── result.html               # Result page template
├── app.py                        # Flask backend
├── train_model.py                # Model training script (optional)
├── requirements.txt              # Dependencies
└── README.md                     # Project readme
