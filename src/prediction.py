import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
model = joblib.load(MODEL_PATH)

def predict_customer(customer_data):
    probability = model.predict_proba(customer_data)[0][1]
    prediction = int(probability >= 0.5)
    risk = "High" if probability >= 0.70 else "Medium" if probability >= 0.40 else "Low"
    return {
        "prediction": "Likely to Churn" if prediction else "Not Likely to Churn",
        "churn_probability": round(probability * 100, 2),
        "risk_level": risk
    }
