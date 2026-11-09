import pandas as pd

def classify_customers(data):
    if "Days_Since_Last_Purchase" not in data.columns:
        raise ValueError("Required column 'Days_Since_Last_Purchase' is missing.")

    days = pd.to_numeric(data["Days_Since_Last_Purchase"], errors="coerce")

    data = data.copy()
    data["Customer_Status"] = "Unknown"
    data.loc[days.between(0, 30), "Customer_Status"] = "Active"
    data.loc[days.between(31, 60), "Customer_Status"] = "Cooling Down"
    data.loc[days.between(61, 90), "Customer_Status"] = "Inactive"
    data.loc[days > 90, "Customer_Status"] = "Dormant"

    return data

def get_activity_summary(data):
    counts = data["Customer_Status"].value_counts()

    return {
        "active": int(counts.get("Active", 0)),
        "cooling_down": int(counts.get("Cooling Down", 0)),
        "inactive": int(counts.get("Inactive", 0)),
        "dormant": int(counts.get("Dormant", 0)),
        "unknown": int(counts.get("Unknown", 0))
    }
