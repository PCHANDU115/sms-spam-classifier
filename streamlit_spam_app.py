import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from catboost import CatBoostClassifier

# Inject custom CSS for additional styling
def inject_custom_css(theme):
    if theme == "Dark":
        st.markdown(
            """
            <style>
            body {
                background-color: #121212;
                color: white;
            }
            .stButton>button {
                background-color: #444;
                color: white;
                border-radius: 12px;
                font-size: 16px;
                padding: 10px 20px;
            }
            .stTextInput>div>input {
                border: 2px solid #888;
                padding: 10px;
                border-radius: 5px;
                color: white;
                background-color: #333;
            }
            .stSelectbox>div>div>button {
                background-color: #555;
                color: white;
                font-size: 14px;
            }
            .stFileUploader {
                border: 2px dashed #888;
                border-radius: 8px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            body {
                background-color: #f7f7f7;
                color: black;
            }
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                border-radius: 12px;
                font-size: 16px;
                padding: 10px 20px;
            }
            .stTextInput>div>input {
                border: 2px solid #4CAF50;
                padding: 10px;
                border-radius: 5px;
            }
            .stSelectbox>div>div>button {
                background-color: #0066cc;
                color: white;
                font-size: 14px;
            }
            .stFileUploader {
                border: 2px dashed #4CAF50;
                border-radius: 8px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

# Add a theme selector
theme = st.sidebar.radio("🎨 Select Theme", ["Light", "Dark"])
inject_custom_css(theme)

# Placeholder code for training and prediction
st.title("📱 SMS Spam Detection Web App")
st.sidebar.title("🌟 Navigation")
option = st.sidebar.selectbox("Choose an action", ["Train Models", "Predict SMS"])

if option == "Train Models":
    st.header("🔧 Train Models")
    uploaded_file = st.file_uploader("📂 Upload a CSV file with 'text' and 'label' columns", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
elif option == "Predict SMS":
    st.header("📩 Predict SMS")
    message = st.text_input("✏️ Enter an SMS message:")
    model_name = st.selectbox("🛠️ Choose a model", ["SVM", "CatBoost"])
    if st.button("🔍 Predict"):
        st.success(f"📋 Prediction: Ham or Spam")
