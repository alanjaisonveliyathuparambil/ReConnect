import pandas as pd

def prioritize_customers(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()

    probability = pd.to_numeric(
        result["Churn_Probability"], errors="coerce"
    ).fillna(0)

    if probability.max() <= 1:
        probability = probability * 100

    probability = probability.clip(0, 100)

    result["Risk_Level"] = probability.map(
        lambda score: "High" if score >= 70
        else "Medium" if score >= 40
        else "Low"
    )

    indicators = pd.to_numeric(
        result["Indicator_Count"], errors="coerce"
    ).fillna(0)

    indicator_score = (indicators / 5 * 100).clip(0, 100)

    lifetime_value = pd.to_numeric(
        result["Lifetime_Value"], errors="coerce"
    ).fillna(0).clip(lower=0)

    max_value = lifetime_value.max()

    if max_value > 0:
        value_score = lifetime_value / max_value * 100
    else:
        value_score = pd.Series(0, index=result.index)

    result["Priority_Score"] = (
        probability * 0.70
        + indicator_score * 0.20
        + value_score * 0.10
    ).clip(0, 100).round(2)

    result["Priority_Level"] = result["Priority_Score"].map(
        lambda score: "High" if score >= 70
        else "Medium" if score >= 40
        else "Low"
    )

    result = result.sort_values(
        "Priority_Score", ascending=False
    )

    return result