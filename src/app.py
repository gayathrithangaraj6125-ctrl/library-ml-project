import streamlit as st
import joblib
import numpy as np
from datetime import datetime

# Load trained model
model = joblib.load("model/library_model.pkl")

st.title("📚 Library Book Rating Prediction")

st.write("Enter book details to predict its Rating")

# Input fields based on dataset features
pages = st.number_input("Number of Pages (pagesNumber)", 10, 2000, 300)
reviews = st.number_input("Number of Reviews (CountsOfReview)", 0, 100000, 1000)
publish_year = st.number_input("Publish Year (PublishYear)", 1900, datetime.now().year, 2015)

# Feature Engineering (same as training)
current_year = datetime.now().year
book_age = current_year - publish_year

if st.button("Predict Rating"):
    features = np.array([[pages, reviews, book_age]])
    prediction = model.predict(features)
    
    st.success(f"Predicted Book Rating: {round(prediction[0], 2)} ⭐")
