from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.data_preparation import prepare_data

def compare_models():
    X_train, X_test, y_train, y_test, _ = prepare_data()
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    print("\nMODEL COMPARISON")
    print("-" * 85)
    print(f"{'Model':<22}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1':<12}{'ROC-AUC'}")
    print("-" * 85)
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        print(f"{name:<22}{accuracy_score(y_test, predictions):<12.4f}{precision_score(y_test, predictions):<12.4f}{recall_score(y_test, predictions):<12.4f}{f1_score(y_test, predictions):<12.4f}{roc_auc_score(y_test, probabilities):.4f}")

if __name__ == "__main__":
    compare_models()
