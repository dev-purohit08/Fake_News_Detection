import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Load the trained pipeline
model = joblib.load("model1.pkl")

# Preprocessing function (optional; pipeline already handles vectorization)
def preprocess(text):
    text = re.sub(r'\W', ' ', text.lower())
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Streamlit UI
st.title("Fake News Detection App")
st.write("Enter a news article and check if it is Fake or Real.")

news_text = st.text_area("Enter the news text here:")

if st.button("Predict"):
    if news_text.strip():
        processed_text = preprocess(news_text)
        # Pipeline handles vectorization and model prediction
        prediction = model.predict([processed_text])[0]
        if prediction == 0:
            st.error("Prediction: Fake News")
        else:
            st.success("Prediction: Real News")
    else:
        st.warning("Please enter some text!")
