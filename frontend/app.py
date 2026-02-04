import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import os
import plotly.express as px

# ---------------- Page Config ----------------
st.set_page_config(page_title="Loan Approval Portal", layout="wide", page_icon="🏦")

# ---------------- Sidebar ----------------
st.sidebar.markdown("""
<div style='background-color:#6A1B9A; padding:15px; border-radius:10px; text-align:center;'>
    <h2 style='color:white;'>🏦 Loan Portal</h2>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("**Instructions:**\n- Use the forms to train, test, or predict loan status.\n- Fill details according to the guidelines below.")
st.sidebar.markdown("---")

# User name
user_name = st.sidebar.text_input("Enter your name:", "Guest")

# ---------------- History Setup ----------------
history_file = "prediction_history.csv"
if not os.path.exists(history_file):
    df_hist = pd.DataFrame(columns=["Timestamp", "Name", "Prediction", "Reasons"])
    df_hist.to_csv(history_file, index=False)
else:
    df_hist = pd.read_csv(history_file)

# ---------------- Header ----------------
st.markdown("""
<div style='background: linear-gradient(90deg, #FF6F61, #6A1B9A);
            padding: 20px; border-radius: 10px; text-align: center;'>
    <h1 style='color: white;'>🏦 Loan Approval Predictor</h1>
    <p style='color: #FFE0B2;'>Train, Test, and Predict Loan Status Instantly!</p>
</div>
""", unsafe_allow_html=True)

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["Train Model", "Test Model", "Predict Loan"])

# ---------------- TRAIN ----------------
with tab1:
    st.write("### Upload Training CSV")
    train_file = st.file_uploader("Upload CSV for Training", type=["csv"], key="train")
    if train_file:
        if st.button("Train Model"):
            try:
                response = requests.post("http://127.0.0.1:8000/train", files={"file": train_file})
                result = response.json()
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success("✅ Model trained successfully!")
                    st.write(f"**Accuracy:** {result['accuracy']}")
                    st.write("**Classification Report:**")
                    st.json(result["classification_report"])
            except Exception as e:
                st.error(f"Connection Error: {e}")

# ---------------- TEST ----------------
with tab2:
    st.write("### Upload Testing CSV")
    test_file = st.file_uploader("Upload CSV for Testing", type=["csv"], key="test")
    if test_file:
        if st.button("Test Model"):
            try:
                response = requests.post("http://127.0.0.1:8000/test", files={"file": test_file})
                result = response.json()
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success("✅ Testing Completed!")
                    st.write(f"**Accuracy:** {result['accuracy']}")
                    st.write("**Classification Report:**")
                    st.json(result["classification_report"])
            except Exception as e:
                st.error(f"Connection Error: {e}")

# ---------------- PREDICT ----------------
with tab3:
    st.write("### Enter Your Details for Prediction")

    # ---------------- Field Guidelines ----------------
    field_guidelines = {
        "no_of_dependents": "Number of dependents (0-10)",
        "education": "Education: Graduate or Not Graduate",
        "self_employed": "Are you self-employed? Yes or No",
        "income_annum": "Annual income in INR (e.g., 5000000)",
        "loan_amount": "Loan amount requested in INR",
        "loan_term": "Loan term in months",
        "cibil_score": "CIBIL score (300-900)",
        "residential_assets_value": "Value of residential assets in INR",
        "commercial_assets_value": "Value of commercial assets in INR",
        "luxury_assets_value": "Value of luxury assets in INR",
        "bank_asset_value": "Value of bank assets in INR"
    }

    with st.form("loan_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            no_of_dependents = st.number_input(
                f"Number of Dependents ({field_guidelines['no_of_dependents']})",
                0, 10, step=1
            )
            education = st.selectbox(
                f"Education ({field_guidelines['education']})",
                ["Graduate", "Not Graduate"]
            )
            self_employed = st.selectbox(
                f"Self Employed ({field_guidelines['self_employed']})",
                ["Yes", "No"]
            )
            income_annum = st.number_input(
                f"Annual Income (₹) ({field_guidelines['income_annum']})", 0, step=5000
            )

        with col2:
            loan_amount = st.number_input(
                f"Loan Amount (₹) ({field_guidelines['loan_amount']})", 0, step=5000
            )
            loan_term = st.number_input(
                f"Loan Term (months) ({field_guidelines['loan_term']})", 1, 600
            )
            cibil_score = st.slider(
                f"CIBIL Score ({field_guidelines['cibil_score']})", 300, 900, 700, step=1
            )

        with col3:
            residential_assets_value = st.number_input(
                f"Residential Assets (₹) ({field_guidelines['residential_assets_value']})", 0, step=10000
            )
            commercial_assets_value = st.number_input(
                f"Commercial Assets (₹) ({field_guidelines['commercial_assets_value']})", 0, step=10000
            )
            luxury_assets_value = st.number_input(
                f"Luxury Assets (₹) ({field_guidelines['luxury_assets_value']})", 0, step=10000
            )
            bank_asset_value = st.number_input(
                f"Bank Assets (₹) ({field_guidelines['bank_asset_value']})", 0, step=10000
            )

        submitted = st.form_submit_button("Predict Loan Status")

    if submitted:
        data = {
            "no_of_dependents": no_of_dependents,
            "education": education,
            "self_employed": self_employed,
            "income_annum": income_annum,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "cibil_score": cibil_score,
            "residential_assets_value": residential_assets_value,
            "commercial_assets_value": commercial_assets_value,
            "luxury_assets_value": luxury_assets_value,
            "bank_asset_value": bank_asset_value
        }

        try:
            response = requests.post("http://127.0.0.1:8000/predict", json=data)
            result = response.json()
            st.markdown("---")

            # ---------------- Save Prediction ----------------
            new_row = pd.DataFrame([{
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Name": user_name,
                "Prediction": result.get("prediction", "Error"),
                "Reasons": result.get("reason", "")
            }])
            df_hist = pd.concat([df_hist, new_row], ignore_index=True)
            df_hist.to_csv(history_file, index=False)

            # ---------------- Result Display ----------------
            if result.get("prediction") == "Approved":
                st.success("✅ Loan Approved! Congratulations!")
            else:
                st.error("❌ Loan Rejected")
                if result.get("reason"):
                    st.markdown("**Reasons for Rejection:**")
                    for reason in result["reason"].split(", "):
                        st.markdown(f"- {reason}")

            # ---------------- Metrics ----------------
            approved_count = df_hist["Prediction"].value_counts().get("Approved", 0)
            rejected_count = df_hist["Prediction"].value_counts().get("Rejected", 0)

            col1, col2 = st.columns(2)
            col1.metric("Approved Loans", approved_count)
            col2.metric("Rejected Loans", rejected_count)

            # ---------------- Pie Chart ----------------
            fig = px.pie(
                df_hist,
                names="Prediction",
                title="Loan Approval Distribution",
                color_discrete_sequence=["green", "red"]
            )
            st.plotly_chart(fig, use_container_width=True)

            # ---------------- Sidebar History ----------------
            st.sidebar.markdown("### Recent Predictions")
            for i, row in df_hist.tail(5).iterrows():
                st.sidebar.markdown(f"{row['Timestamp']}: **{row['Name']}** - {row['Prediction']}")

        except Exception as e:
            st.error(f"Connection Error: {e}")

# ---------------- Footer ----------------
st.markdown("""
<div style='text-align:center; margin-top:20px; color:#9E9E9E;'>
    Developed by Tanmay Margale | Internship Project at CodeSpyder Technologies
</div>
""", unsafe_allow_html=True)