import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from src.data_preparation import prepare_data

def train_final_model():
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(n_estimators=100, random_state=42))
    ])
    from src.data_loader import load_customer_data
    data = load_customer_data().dropna(subset=["Churned"])
    X = data.drop(columns=["Churned"])
    y = data["Churned"]
    model.fit(X, y)
    joblib.dump(model, "models/churn_model.pkl")
    print("Final model trained and saved successfully!")
    print("Model: Gradient Boosting")
    print("Saved at: models/churn_model.pkl")

if __name__ == "__main__":
    train_final_model()
