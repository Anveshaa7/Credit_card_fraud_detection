import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Title
st.title("Credit Card Fraud Detection")

# File upload
uploaded_file = st.file_uploader("Upload your credit_data.csv", type=["csv"])
if uploaded_file is not None:
    credit_card_data = pd.read_csv(uploaded_file)
    
    # Show overview
    st.subheader("Dataset Overview")
    st.write("First 5 rows:")
    st.dataframe(credit_card_data.head())

    st.write("Last 5 rows:")
    st.dataframe(credit_card_data.tail())

    st.subheader("Data Info")
    st.text(str(credit_card_data.info()))

    # Split dataset
    legit = credit_card_data[credit_card_data.Class == 0]
    fraud = credit_card_data[credit_card_data.Class == 1]
    
    # Balance classes
    legit_sample = legit.sample(n=492, random_state=1)
    new_dataset = pd.concat([legit_sample, fraud], axis=0).sample(frac=1, random_state=2).reset_index(drop=True)

    st.subheader("Balanced Class Distribution")
    st.bar_chart(new_dataset['Class'].value_counts())

    # Features and labels
    X = new_dataset.drop(columns='Class', axis=1)
    Y = new_dataset['Class']

    # Train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)

    # Accuracy display
    st.subheader("Model Accuracy")
    st.write(f"Training Accuracy: {accuracy_score(model.predict(X_train), Y_train):.4f}")
    st.write(f"Testing Accuracy: {accuracy_score(model.predict(X_test), Y_test):.4f}")

    # Prediction form
    st.subheader("🔎 Predict Fraud for a New Transaction")

    input_data = {}
    for col in X.columns:
        input_data[col] = st.number_input(f"{col}", value=float(X[col].mean()))

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0][1]
        if prediction == 1:
            st.error(f"⚠️ The transaction is predicted to be **FRAUD** (Probability: {prediction_proba:.2%})")
        else:
            st.success(f"✅ The transaction is predicted to be **LEGITIMATE** (Probability: {1 - prediction_proba:.2%})")
