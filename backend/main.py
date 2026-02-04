from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/loan_model.pkl"

# ---------------- Pydantic model for prediction ----------------
class LoanPredictionInput(BaseModel):
    no_of_dependents: int = Field(..., ge=0, le=10, description="Number of dependents (0-10)")
    education: str = Field(..., description="Education: Graduate or Not Graduate")
    self_employed: str = Field(..., description="Self employed? Yes or No")
    income_annum: float = Field(..., ge=0, description="Annual income in INR")
    loan_amount: float = Field(..., ge=0, description="Loan amount requested in INR")
    loan_term: int = Field(..., ge=0, description="Loan term in months")
    cibil_score: int = Field(..., ge=0, le=900, description="CIBIL score (0-900)")
    residential_assets_value: float = Field(..., ge=0, description="Value of residential assets")
    commercial_assets_value: float = Field(..., ge=0, description="Value of commercial assets")
    luxury_assets_value: float = Field(..., ge=0, description="Value of luxury assets")
    bank_asset_value: float = Field(..., ge=0, description="Value of bank assets")


# ------------------ Utility function: preprocess dataset ------------------
def preprocess(df: pd.DataFrame):
    df.columns = df.columns.str.strip()
    if 'loan_id' in df.columns:
        df.drop(['loan_id'], axis=1, inplace=True)

    num_cols = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term',
                'cibil_score', 'residential_assets_value', 'commercial_assets_value',
                'luxury_assets_value', 'bank_asset_value']
    for col in num_cols:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)

    cat_cols = ['education', 'self_employed', 'loan_status']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.capitalize()

    df.replace({
        'education': {'Graduate':1, 'Not graduate':0},
        'self_employed': {'Yes':1, 'No':0},
        'loan_status': {'Y':1, 'N':0, 'Approved':1, 'Rejected':0}
    }, inplace=True)

    return df

# ------------------ Train Endpoint ------------------
@app.post("/train")
async def train_model(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        df = preprocess(df)

        X = df.drop('loan_status', axis=1)
        y = df['loan_status']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        return {
            "message": "Model trained and saved successfully!",
            "accuracy": round(acc,4),
            "classification_report": report
        }
    except Exception as e:
        return {"error": str(e)}


# ------------------ Test Endpoint ------------------
@app.post("/test")
async def test_model(file: UploadFile = File(...)):
    try:
        if not os.path.exists(MODEL_PATH):
            return {"error": "Model not found. Train first."}

        model = joblib.load(MODEL_PATH)
        df = pd.read_csv(file.file)
        df = preprocess(df)

        X = df.drop('loan_status', axis=1)
        y = df['loan_status']

        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, output_dict=True)

        return {
            "accuracy": round(acc,4),
            "classification_report": report
        }
    except Exception as e:
        return {"error": str(e)}

# ------------------ Predict Endpoint ------------------
@app.post("/predict")
async def predict(data: LoanPredictionInput):
    try:
        if not os.path.exists(MODEL_PATH):
            return {"error": "Model not found. Train first."}

        model = joblib.load(MODEL_PATH)
        df = pd.DataFrame([data.dict()])

        # Preprocess categorical columns
        cat_cols = ['education','self_employed']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.capitalize()
        df.replace({
            'education': {'Graduate':1, 'Not graduate':0},
            'self_employed': {'Yes':1, 'No':0}
        }, inplace=True)

        # Ensure numeric columns
        num_cols = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term',
                    'cibil_score', 'residential_assets_value', 'commercial_assets_value',
                    'luxury_assets_value', 'bank_asset_value']
        for col in num_cols:
            if col not in df.columns:
                df[col] = 0

        pred = model.predict(df)[0]
        result = "Approved" if pred==1 else "Rejected"

        reason = ""
        if result == "Rejected":
            reasons = []
            if df.loc[0, "cibil_score"] < 600:
                reasons.append("Low CIBIL score")
            if df.loc[0, "income_annum"] < df.loc[0, "loan_amount"]*0.3:
                reasons.append("Income too low for loan amount")
            if df.loc[0, "no_of_dependents"] > 4:
                reasons.append("High number of dependents")
            if df.loc[0, "education"] == 0:  # Not Graduate
                reasons.append("Education level may affect approval")
            reason = ", ".join(reasons) if reasons else "Other risk factors"

        return {"prediction": result, "reason": reason}

    except Exception as e:
        return {"error": str(e)}