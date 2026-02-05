import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import os
import plotly.express as px

# ================= CONFIG =================
BACKEND_URL = "https://loanapprovalprediction-3h7m.onrender.com"

st.set_page_config(page_title="Loan Approval Portal", layout="wide", page_icon="🏦")

# ================= SIDEBAR =================
st.sidebar.markdown("""
<div style='background-color:#6A1B9A; padding:15px; border-radius:10px; text-align:center;'>
<h2 style='color:white;'>🏦 Loan Portal</h2>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
**Instructions**
- Upload dataset to train/test
- Fill details to predict
""")

user_name = st.sidebar.text_input("Enter your name:", "Guest")

# ================= HISTORY =================
history_file = "prediction_history.csv"

if not os.path.exists(history_file):
    pd.DataFrame(columns=["Timestamp", "Name", "Prediction", "Reasons"]).to_csv(history_file, index=False)

df_hist = pd.read_csv(history_file)

# ================= HEADER =================
st.markdown("""
<div style='background: linear-gradient(90deg,#FF6F61,#6A1B9A);
padding:20px;border-radius:10px;text-align:center;'>
<h1 style='color:white;'>🏦 Loan Approval Predictor</h1>
<p style='color:#FFE0B2;'>Train • Test • Predict Instantly</p>
</div>
""", unsafe_allow_html=True)

# ================= TABS =================
tab1, tab2, tab3 = st.tabs(["Train Model", "Test Model", "Predict Loan"])

# =====================================================
# TRAIN
# =====================================================
with tab1:
    st.subheader("Upload Training CSV")

    train_file = st.file_uploader("Training dataset", type="csv")

    if train_file and st.button("Train Model"):

        try:
            res = requests.post(
                f"{BACKEND_URL}/train",
                files={"file": train_file}
            )

            result = res.json()

            st.success("Model trained!")

            st.write("Accuracy:", result.get("accuracy"))
            st.json(result.get("classification_report"))

        except Exception as e:
            st.error(f"Connection error: {e}")

# =====================================================
# TEST
# =====================================================
with tab2:
    st.subheader("Upload Testing CSV")

    test_file = st.file_uploader("Testing dataset", type="csv")

    if test_file and st.button("Test Model"):

        try:
            res = requests.post(
                f"{BACKEND_URL}/test",
                files={"file": test_file}
            )

            result = res.json()

            st.success("Testing complete!")

            st.write("Accuracy:", result.get("accuracy"))
            st.json(result.get("classification_report"))

        except Exception as e:
            st.error(f"Connection error: {e}")

# =====================================================
# PREDICT
# =====================================================
with tab3:

    st.subheader("Enter Loan Details")

    with st.form("loan_form"):

        col1, col2, col3 = st.columns(3)

        with col1:
            dependents = st.number_input("Dependents", 0, 10)
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_emp = st.selectbox("Self Employed", ["Yes", "No"])
            income = st.number_input("Annual Income", 0)

        with col2:
            loan_amt = st.number_input("Loan Amount", 0)
            loan_term = st.number_input("Loan Term (months)", 1, 600)
            cibil = st.slider("CIBIL Score", 300, 900, 700)

        with col3:
            res_asset = st.number_input("Residential Assets", 0)
            com_asset = st.number_input("Commercial Assets", 0)
            lux_asset = st.number_input("Luxury Assets", 0)
            bank_asset = st.number_input("Bank Assets", 0)

        submit = st.form_submit_button("Predict")

    if submit:

        payload = {
            "no_of_dependents": dependents,
            "education": education,
            "self_employed": self_emp,
            "income_annum": income,
            "loan_amount": loan_amt,
            "loan_term": loan_term,
            "cibil_score": cibil,
            "residential_assets_value": res_asset,
            "commercial_assets_value": com_asset,
            "luxury_assets_value": lux_asset,
            "bank_asset_value": bank_asset
        }

        try:
            res = requests.post(f"{BACKEND_URL}/predict", json=payload)
            result = res.json()

            prediction = result.get("prediction", "Error")
            reasons = result.get("reasons", [])

            # Save history
            new_row = pd.DataFrame([{
                "Timestamp": datetime.now(),
                "Name": user_name,
                "Prediction": prediction,
                "Reasons": ", ".join(reasons)
            }])

            df_hist = pd.concat([df_hist, new_row], ignore_index=True)
            df_hist.to_csv(history_file, index=False)

            # Display result
            st.markdown("---")

            if prediction == "Approved":
                st.success("Loan Approved!")
            else:
                st.error("Loan Rejected")

                for r in reasons:
                    st.write("•", r)

            # Metrics
            col1, col2 = st.columns(2)
            col1.metric("Approved", (df_hist["Prediction"] == "Approved").sum())
            col2.metric("Rejected", (df_hist["Prediction"] == "Rejected").sum())

            # Pie chart
            fig = px.pie(df_hist, names="Prediction", title="Loan Distribution")
            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Connection error: {e}")

# ================= FOOTER =================
st.markdown("""
<div style='text-align:center;margin-top:20px;color:#9E9E9E;'>
Developed by Tanmay Margale
</div>
""", unsafe_allow_html=True)