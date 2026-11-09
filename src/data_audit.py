from src.data_loader import load_customer_data

def audit_dataset():
    data = load_customer_data()

    print("\nDATASET OVERVIEW")
    print("Shape:", data.shape)
    print("Columns:", data.columns.tolist())

    print("\nDATA TYPES")
    print(data.dtypes)

    print("\nMISSING VALUES")
    print(data.isnull().sum())

    print("\nDUPLICATE ROWS")
    print(data.duplicated().sum())

    if "Churned" in data.columns:
        print("\nCHURN DISTRIBUTION")
        print(data["Churned"].value_counts(dropna=False))
        print("\nCHURN PERCENTAGE")
        print(data["Churned"].value_counts(normalize=True, dropna=False) * 100)
    else:
        print("\nERROR: Churned column is missing.")

if __name__ == "__main__":
    audit_dataset()
