# SMS Spam Detection App

## Purpose
The **SMS Spam Detection App** uses machine learning models to classify SMS messages as either **Spam** or **Ham**. The app allows users to upload a dataset, train two models (**SVM** and **CatBoost**), and then use the trained models to predict whether a new SMS message is spam or not.

This project aims to provide a simple way for users to detect spam messages in their SMS inbox using machine learning techniques.

## Requirements
To run this project, you need to install the following libraries:
- **streamlit**: For creating the web application.
- **pandas**: For data manipulation.
- **scikit-learn**: For machine learning models.
- **catboost**: For training the CatBoost model.
- **pickle**: For saving and loading models.

## Steps to Implement the Model
1. Dataset Preprocessing
The dataset used in this project should have two columns:

text: Contains the SMS messages.
label: Contains labels (either "spam" or "ham").
Preprocessing Steps:
Loading the Dataset: The user is prompted to upload a CSV file containing the SMS data.
Data Cleaning: Remove any unnecessary columns or rows with missing values.
Label Encoding: Convert the labels ('ham' and 'spam') into numeric values (0 for ham, 1 for spam).
Text Vectorization: Use TF-IDF (Term Frequency-Inverse Document Frequency) to convert text data into numerical features.
2. Model Training
Two machine learning models are used for spam classification:

SVM (Support Vector Machine): A linear kernel SVM is trained to classify SMS messages.
CatBoost: A gradient boosting algorithm is also trained on the same dataset.
The models are trained and evaluated using the following steps:

Splitting the data: Split the data into training (80%) and testing (20%) sets.
Model Evaluation: After training, the accuracy on the training and test sets is displayed for both models.
3. Model Deployment
Once the models are trained, they are saved into pickle files for future use:

vectorizer.pkl: The trained TF-IDF vectorizer.
svm_model.pkl: The trained SVM model.
catboost_model.pkl: The trained CatBoost model.
These models are then used to predict whether new SMS messages are spam or ham.

4. Prediction
Users can enter a new SMS message and select which model to use for classification (SVM or CatBoost).
The message is vectorized using the saved TF-IDF vectorizer.
The selected model classifies the message as Spam or Ham and displays the result along with the model’s test accuracy.
