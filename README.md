# 🏦 Loan Approval Prediction Portal

**Internship Project – CodeSpyder Technologies Pvt. Ltd.**  
**Developed by:** Tanmay Margale  

---

## **Project Overview**

This project predicts whether a loan application will be **Approved** or **Rejected** based on applicant financial and personal details.  
It provides a **user-friendly Streamlit interface** and a **FastAPI backend** for training, testing, and predicting using machine learning.  

**Key Features:**
- Train ML model with a CSV dataset using `/train` endpoint  
- Test ML model with a CSV dataset using `/test` endpoint  
- Predict loan approval for a single applicant using `/predict` endpoint  
- Shows **reason for rejection** if loan is rejected  
- Maintains **prediction history** and visualizes loan approval distribution  
- Frontend developed using **Streamlit** with field guidance for user inputs  

---

## **Dataset**

- Dataset used: `loan_data.csv`  
- Features:
  | Column | Description |
  |--------|-------------|
  | no_of_dependents | Number of dependents |
  | education | Graduate / Not Graduate |
  | self_employed | Yes / No |
  | income_annum | Annual income in INR |
  | loan_amount | Requested loan amount in INR |
  | loan_term | Loan term in months |
  | cibil_score | CIBIL score (300-900) |
  | residential_assets_value | Value of residential assets in INR |
  | commercial_assets_value | Value of commercial assets in INR |
  | luxury_assets_value | Value of luxury assets in INR |
  | bank_asset_value | Value of bank assets in INR |
  | loan_status | Approved / Rejected (target) |

> You can find sample datasets in the `dataset/` folder.

---

## **Folder Structure**
ML_WebApp/
├── backend/ # FastAPI backend code
│ ├── main.py
│ └── requirements.txt
├── frontend/ # Streamlit frontend code
│ └── app.py
├── dataset/ # Sample training/testing datasets
│ ├── loan_train.csv
│ └── loan_test.csv
├── models/ # Saved ML models
│ └── loan_model.pkl
├── prediction_history.csv # Stores prediction history
├── demo_video.mp4 # Demo video of project
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## **Setup Instructions**

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/LoanApprovalPrediction.git
cd LoanApprovalPrediction

python -m venv myenv
# Windows
myenv\Scripts\activate
# Linux/Mac
source myenv/bin/activate

pip install -r requirements.txt

cd backend
uvicorn main:app --reload

cd frontend
streamlit run app.py

Usage
1. Train Model

Upload a CSV dataset in the Train tab.

Click Train Model.

Model will be trained and saved in models/loan_model.pkl.

Accuracy and classification report will be displayed.

2. Test Model

Upload a CSV dataset in the Test tab.

Click Test Model.

Test results will be displayed with accuracy and classification report.

3. Predict Loan

Fill the form in the Predict tab with applicant details.

Click Predict Loan Status.

Result will show Approved or Rejected with reason(s) if rejected.

Recent predictions and metrics are visualized with charts.

Deployment

Frontend can be deployed using Streamlit Community Cloud
.

Backend can be deployed using Heroku / Railway / Render for public API access.

Update API URL in app.py to the deployed backend URL before deploying Streamlit.

Technologies Used

Python, FastAPI, Streamlit

Pandas, NumPy, scikit-learn, joblib

Plotly for visualization

Requests for API communication

Author

Tanmay Margale
Final Year, Electronics & Telecommunication Engineering
