import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from catboost import CatBoostClassifier

# Theme Selector
theme = st.sidebar.radio("🎨 Select Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown(
        """
        <style>
        .main {
            background-color: #121212;
            color: white;
        }
        .stApp {
            background-color: #121212;
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
        .main {
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

# Global variables
svm_model = None
catboost_model = None
vectorizer = None

# Function to train the models
def train_models(data):
    global svm_model, catboost_model, vectorizer

    # Preprocess data
    X = data["text"]
    y = data["label"].map({"ham": 0, "spam": 1})  # Map labels to 0 and 1

    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vectorized = vectorizer.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

    # Train SVM model
    svm_model = SVC(kernel="linear", probability=True)
    svm_model.fit(X_train, y_train)
    svm_train_acc = svm_model.score(X_train, y_train)
    svm_test_acc = svm_model.score(X_test, y_test)

    # Train CatBoost model
    catboost_model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, silent=True)
    catboost_model.fit(X_train, y_train)
    catboost_train_acc = catboost_model.score(X_train, y_train)
    catboost_test_acc = catboost_model.score(X_test, y_test)

    # Save models and vectorizer
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open("svm_model.pkl", "wb") as f:
        pickle.dump(svm_model, f)
    with open("catboost_model.pkl", "wb") as f:
        pickle.dump(catboost_model, f)

    return {
        "SVM Train Accuracy": svm_train_acc,
        "SVM Test Accuracy": svm_test_acc,
        "CatBoost Train Accuracy": catboost_train_acc,
        "CatBoost Test Accuracy": catboost_test_acc,
    }

# Load models and vectorizer
def load_models():
    global vectorizer, svm_model, catboost_model
    try:
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        with open("svm_model.pkl", "rb") as f:
            svm_model = pickle.load(f)
        with open("catboost_model.pkl", "rb") as f:
            catboost_model = pickle.load(f)
    except FileNotFoundError:
        st.warning("Models not found! Please train them first.")

# Predict a single message
def predict_message(message, model_name):
    message_vectorized = vectorizer.transform([message])
    if model_name == "SVM":
        prediction = svm_model.predict(message_vectorized)[0]
        return "Spam" if prediction == 1 else "Ham"
    elif model_name == "CatBoost":
        prediction = catboost_model.predict(message_vectorized)[0]
        return "Spam" if prediction == 1 else "Ham"
    else:
        return "Invalid model selected!"

# Streamlit app UI
st.title("📱 SMS Spam Detection Web App")
st.sidebar.title("🌟 Navigation")
option = st.sidebar.selectbox("Choose an action", ["Train Models", "Predict SMS"])

if option == "Train Models":
    st.header("🔧 Train Models")
    uploaded_file = st.file_uploader("📂 Upload a CSV file with 'text' and 'label' columns", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        if "text" in data.columns and "label" in data.columns:
            with st.spinner("Training models... Please wait."):
                result = train_models(data)
            st.success("✅ Models trained successfully!")
            st.json(result)
        else:
            st.error("❌ The uploaded file must contain 'text' and 'label' columns.")
elif option == "Predict SMS":
    st.header("📩 Predict SMS")
    load_models()
    message = st.text_input("✏️ Enter an SMS message:")
    model_name = st.selectbox("🛠️ Choose a model", ["SVM", "CatBoost"])
    if st.button("🔍 Predict"):
        if vectorizer and (svm_model or catboost_model):
            with st.spinner("Predicting... Please wait."):
                prediction = predict_message(message, model_name)
            st.success(f"📋 The message is classified as: **{prediction}**")
        else:
            st.error("❌ Models not loaded! Please train them first.")
