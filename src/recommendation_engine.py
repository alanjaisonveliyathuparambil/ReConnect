import pandas as pd

def generate_recommendations(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()

    days = pd.to_numeric(
        result["Days_Since_Last_Purchase"], errors="coerce"
    ).fillna(0)

    logins = pd.to_numeric(
        result["Login_Frequency"], errors="coerce"
    ).fillna(0)

    cart = pd.to_numeric(
        result["Cart_Abandonment_Rate"], errors="coerce"
    ).fillna(0)

    email = pd.to_numeric(
        result["Email_Open_Rate"], errors="coerce"
    ).fillna(0)

    returns = pd.to_numeric(
        result["Returns_Rate"], errors="coerce"
    ).fillna(0)

    cart = cart.where(cart <= 1, cart / 100)
    email = email.where(email <= 1, email / 100)
    returns = returns.where(returns <= 1, returns / 100)

    recommendations = []

    for customer in zip(days, logins, cart, email, returns):
        days_value, login_value, cart_value, email_value, return_value = customer

        actions = []

        if days_value >= 90:
            actions.append("Send personalized comeback offer")
        elif days_value >= 30:
            actions.append("Send purchase reminder")

        if login_value < 3:
            actions.append("Send re-engagement email")

        if cart_value >= 0.70:
            actions.append("Offer cart recovery discount")

        if email_value < 0.20:
            actions.append("Try an alternative communication channel")

        if return_value >= 0.30:
            actions.append("Arrange customer support follow-up")

        if not actions:
            actions.append("Maintain regular engagement")

        recommendations.append(actions)

    result["Retention_Recommendations"] = recommendations

    return result