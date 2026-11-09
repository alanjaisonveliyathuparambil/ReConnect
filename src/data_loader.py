from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "data" / "raw" / "customers.csv"

def load_customer_data():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    data = pd.read_csv(DATASET_PATH)

    if data.empty:
        raise ValueError("The dataset is empty.")

    return data
