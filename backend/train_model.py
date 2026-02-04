import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

df = pd.read_csv("../dataset/loan_data.csv")

df.head()
df.info()

print(df.isnull().sum())

df.columns
df.columns = df.columns.str.strip()
df.drop(['loan_id'], axis=1, inplace=True)
print(df.columns.tolist())

num_cols = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term',
            'cibil_score', 'residential_assets_value', 'commercial_assets_value',
            'luxury_assets_value', 'bank_asset_value']

for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

cat_cols = ['education', 'self_employed', 'loan_status']
for col in cat_cols:
    df[col] = df[col].astype(str).str.strip().str.capitalize()  # Capitalize first letter

# Check unique values
for col in cat_cols:
    print(col, df[col].unique())


df.replace({
    'education': {'Graduate':1, 'Not graduate':0},
    'self_employed': {'Yes':1, 'No':0},
    'loan_status': {'Y':1, 'N':0}
}, inplace=True)


X = df.drop('loan_status', axis=1)
y = df['loan_status']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/loan_model.pkl")
print("Model saved successfully!")