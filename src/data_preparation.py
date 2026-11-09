import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.data_loader import load_customer_data

def prepare_data():
    data = load_customer_data()

    data = data.dropna(subset=["Churned"])

    if not set(data["Churned"].unique()).issubset({0, 1}):
        raise ValueError("Churned must contain only 0 and 1.")

    X = data.drop(columns=["Churned"])
    y = data["Churned"]

    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(include=["object", "str", "category"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()

    print("\nDATA PREPARATION COMPLETE")
    print("Training records:", X_train.shape[0])
    print("Testing records:", X_test.shape[0])
    print("Training features:", X_train.shape[1])
    print("Testing features:", X_test.shape[1])
    print("\nTraining target distribution:")
    print(y_train.value_counts())
    print("\nTesting target distribution:")
    print(y_test.value_counts())
