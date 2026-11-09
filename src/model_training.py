import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from src.data_preparation import prepare_data

BASE_DIR = Path(__file__).resolve().parent.parent

def train_model():
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()

    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nGRADIENT BOOSTING RESULTS")
    print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
    print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))
    print("\nCLASSIFICATION REPORT")
    print(classification_report(y_test, predictions))

    model_bundle = {
        "model": model,
        "preprocessor": preprocessor,
        "feature_columns": list(preprocessor.feature_names_in_)
    }

    model_path = BASE_DIR / "models" / "churn_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_bundle, model_path)

    print("\nModel and preprocessor saved successfully.")
    print("Location:", model_path)

    return model_bundle

if __name__ == "__main__":
    train_model()
