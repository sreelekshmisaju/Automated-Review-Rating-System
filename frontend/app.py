import streamlit as st
import joblib
import re
import spacy
import numpy as np
import time

# --- Load NLP Model ---
nlp = spacy.load("en_core_web_sm")

# --- Load Models and Vectorizers ---
model_dir = "../models"
model_A = joblib.load(f"{model_dir}/Model_Balanced.pkl")
model_B = joblib.load(f"{model_dir}/Model_Imbalanced.pkl")
vectorizer_A = joblib.load(f"{model_dir}/tfidf_Balanced.pkl")
vectorizer_B = joblib.load(f"{model_dir}/tfidf_Imbalanced.pkl")

# --- Helper Functions ---
def preprocess_review(text):
    """Clean, remove stopwords, and lemmatize text."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if token.text not in nlp.Defaults.stop_words])

def predict_review(review, model, vectorizer):
    processed = preprocess_review(review)
    tfidf_review = vectorizer.transform([processed])
    prediction = model.predict(tfidf_review)[0]
    confidence = model.predict_proba(tfidf_review).max()
    return prediction, confidence

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Review Rating Predictor", page_icon="⭐", layout="wide")

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
        body {
            background-color: #f8f9fa;
        }
        .main-title {
            text-align: center;
            color: #2E8B57;
            font-size: 40px;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .sub-title {
            text-align: center;
            color: #555;
            font-size: 18px;
            margin-bottom: 40px;
        }
        .card {
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            background-color: white;
            transition: transform 0.3s ease;
        }
        .card:hover {
            transform: scale(1.03);
        }
        .balanced {
            border-left: 8px solid #4CAF50;
        }
        .imbalanced {
            border-left: 8px solid #FF9800;
        }
        .confidence-bar {
            height: 10px;
            border-radius: 10px;
            margin-top: 5px;
        }
        .footer {
            text-align:center; 
            color:gray; 
            margin-top:50px;
            font-size:14px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title Section ---
st.markdown("<div class='main-title'>📊 Review Rating Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Compare predictions from Balanced vs. Imbalanced Models</div>", unsafe_allow_html=True)

# --- Input Section ---
review_input = st.text_area("👩🏻‍💻 Enter a product review below:", placeholder="Type your review here...")

if st.button(" 📊📈Predict Rating"):
    if review_input.strip() == "":
        st.warning("💻Please enter a review first.")
    else:
        with st.spinner("…. Loading Analyzing review......."):
            time.sleep(1.2)
            pred_A, conf_A = predict_review(review_input, model_A, vectorizer_A)
            pred_B, conf_B = predict_review(review_input, model_B, vectorizer_B)

        st.success(" Prediction completed successfully")

        # --- Display Results Side by Side ---
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("<div class='card balanced'>", unsafe_allow_html=True)
            st.markdown("### ⚖️ Model A (Balanced Data)")
            st.write(f"**Predicted Rating:** ⭐ {pred_A}")
            st.progress(float(conf_A))
            st.write(f"**Confidence:** {conf_A:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='card imbalanced'>", unsafe_allow_html=True)
            st.markdown("### 📊 Model B (Imbalanced Data)")
            st.write(f"**Predicted Rating:** ⭐ {pred_B}")
            st.progress(float(conf_B))
            st.write(f"**Confidence:** {conf_B:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)

        

st.markdown("<div class='footer'>Made with ❤️ using Streamlit & SpaCy</div>", unsafe_allow_html=True)
