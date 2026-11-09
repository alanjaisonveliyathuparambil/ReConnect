import pandas as pd

def analyze_engagement(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()

    days = pd.to_numeric(
        result["Days_Since_Last_Purchase"], errors="coerce"
    ).fillna(0)

    logins = pd.to_numeric(
        result["Login_Frequency"], errors="coerce"
    ).fillna(0)

    session = pd.to_numeric(
        result["Session_Duration_Avg"], errors="coerce"
    ).fillna(0)

    cart = pd.to_numeric(
        result["Cart_Abandonment_Rate"], errors="coerce"
    ).fillna(0)

    email = pd.to_numeric(
        result["Email_Open_Rate"], errors="coerce"
    ).fillna(0)

    cart = cart.where(cart <= 1, cart / 100)
    email = email.where(email <= 1, email / 100)

    conditions = [
        days >= 90,
        logins < 3,
        session < 2,
        cart >= 0.70,
        email < 0.20
    ]

    labels = [
        "Long purchase inactivity",
        "Low login frequency",
        "Short session duration",
        "High cart abandonment",
        "Low email engagement"
    ]

    indicators = []

    for values in zip(*[condition.to_numpy() for condition in conditions]):
        indicators.append([
            label for condition, label in zip(values, labels) if condition
        ])

    result["Engagement_Indicators"] = indicators
    result["Indicator_Count"] = result["Engagement_Indicators"].str.len()

    result["Engagement_Level"] = result["Indicator_Count"].map(
        lambda count: "Poor" if count >= 3
        else "Moderate" if count >= 1
        else "Good"
    )

    return result

import joblib
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"

def predict_churn(data):
    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    feature_columns = bundle["feature_columns"]

    result = data.copy()

    X = result.drop(columns=["Churned"], errors="ignore")

    missing_columns = set(feature_columns) - set(X.columns)

    if missing_columns:
        raise ValueError(
            f"Missing customer features: {sorted(missing_columns)}"
        )

    X = X[feature_columns]

    X_processed = preprocessor.transform(X)

    probabilities = model.predict_proba(X_processed)[:, 1]

    result["Churn_Probability"] = probabilities

    result["Churn_Probability_Percentage"] = (
        probabilities * 100
    ).round(2)

    result["Predicted_Churn"] = (
        probabilities >= 0.5
    ).astype(int)

    result["Risk_Level"] = pd.Series(
        probabilities * 100,
        index=result.index
    ).map(
        lambda score: "High" if score >= 70
        else "Medium" if score >= 40
        else "Low"
    )

    return result
