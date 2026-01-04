import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gdown

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Bank Marketing Prediction",
    layout="wide"
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("bank.csv")

data = load_data()

# ---------------- LOAD MODEL ----------------
model = joblib.load("final.pkl")

# 👉 CRITICAL: get trained feature names
trained_features = model.feature_names_in_

# ---------------- SIDEBAR ----------------
st.sidebar.title("🏦 Bank Marketing App")
menu = st.sidebar.radio(
    "Navigate",
    ["Home", "Dataset", "EDA Dashboard", "Prediction"]
)

# ---------------- HOME ----------------
if menu == "Home":
    st.title("📊 Bank Marketing Prediction System")

    st.markdown("""
    ### 🎯 Objective
    Predict whether a customer will **subscribe to a term deposit**.

    ### 🧠 Model Used
    - Logistic Regression / Random Forest  
    - Trained on Bank Marketing Dataset

    ### 🛠 Tools
    - Python
    - Pandas, Matplotlib, Seaborn
    - Scikit-learn
    - Streamlit
    """)

# ---------------- DATASET ----------------
elif menu == "Dataset":
    st.title("📁 Dataset Overview")

    st.subheader("Sample Data")
    st.dataframe(data.head())

    st.subheader("Dataset Shape")
    st.write("Rows:", data.shape[0])
    st.write("Columns:", data.shape[1])

    st.subheader("Target Distribution")
    st.bar_chart(data["deposit"].value_counts())

# ---------------- EDA DASHBOARD ----------------
elif menu == "EDA Dashboard":
    st.title("📈 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Deposit vs Job")
        fig, ax = plt.subplots()
        sns.countplot(y="job", hue="deposit", data=data, ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Deposit vs Education")
        fig, ax = plt.subplots()
        sns.countplot(y="education", hue="deposit", data=data, ax=ax)
        st.pyplot(fig)

    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data["age"], kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Balance vs Deposit")
    fig, ax = plt.subplots()
    sns.boxplot(x="deposit", y="balance", data=data, ax=ax)
    st.pyplot(fig)

# ---------------- PREDICTION ----------------
elif menu == "Prediction":
    st.title("🤖 Customer Deposit Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 100, 30)
        balance = st.number_input("Balance", -5000, 100000, 1000)
        duration = st.number_input("Call Duration (seconds)", 0, 5000, 300)
        campaign = st.number_input("Campaign Contacts", 1, 50, 1)

    with col2:
        job = st.selectbox("Job", sorted(data["job"].unique()))
        education = st.selectbox("Education", sorted(data["education"].unique()))
        housing = st.selectbox("Housing Loan", ["yes", "no"])
        loan = st.selectbox("Personal Loan", ["yes", "no"])

    # ---------------- RAW INPUT ----------------
    input_data = {
        "age": age,
        "balance": balance,
        "duration": duration,
        "campaign": campaign,
        "job": job,
        "education": education,
        "housing": housing,
        "loan": loan
    }

    input_df = pd.DataFrame([input_data])

    # ---------------- UI-SIDE ENCODING ----------------
    input_df = pd.get_dummies(input_df)

    # 🔥 FORCE MATCH WITH TRAINED FEATURES
    input_df = input_df.reindex(columns=trained_features, fill_value=0)

    # ---------------- PREDICTION ----------------
    if st.button("🔍 Predict"):
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.success(f"✅ Customer WILL subscribe (Confidence: {prob*100:.2f}%)")
        else:
            st.error(f"❌ Customer will NOT subscribe (Confidence: {prob*100:.2f}%)")

        st.progress(int(prob * 100))

