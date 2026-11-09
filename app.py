from flask import Flask, render_template, request, send_file
from pathlib import Path
from io import BytesIO

from src.data_loader import load_customer_data
from src.customer_analysis import classify_customers, get_activity_summary
from src.retention_analysis import analyze_engagement, predict_churn
from src.recommendation_engine import generate_recommendations
from src.priority_manager import prioritize_customers

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

def prepare_customer_data():
    data = load_customer_data()
    data = classify_customers(data)

    data.insert(
        0,
        "Customer_Reference",
        [f"REC-{i:05d}" for i in range(1, len(data) + 1)]
    )

    data = predict_churn(data)

    return data

def prepare_retention_data():
    data = prepare_customer_data()
    data = analyze_engagement(data)
    data = generate_recommendations(data)
    data = prioritize_customers(data)

    return data

@app.route("/")
def home():
    summary = {
        "active": 0,
        "cooling_down": 0,
        "inactive": 0,
        "dormant": 0,
        "unknown": 0
    }

    try:
        data = prepare_customer_data()
        summary = get_activity_summary(data)

        total_customers = len(data)
        churned_customers = int(
            (data["Churned"] == 1).sum()
        )
        high_risk_customers = int(
            (data["Risk_Level"] == "High").sum()
        )

        customer_preview = data.head(8).fillna("").to_dict(
            orient="records"
        )

        return render_template(
            "index.html",
            total_customers=total_customers,
            churned_customers=churned_customers,
            high_risk_customers=high_risk_customers,
            customer_preview=customer_preview,
            summary=summary,
            error=None
        )

    except Exception as error:
        return render_template(
            "index.html",
            total_customers=0,
            churned_customers=0,
            high_risk_customers=0,
            customer_preview=[],
            summary=summary,
            error=str(error)
        )

@app.route("/customers")
def customers_page():
    try:
        data = prepare_customer_data()

        search = request.args.get("search", "").strip()
        selected_status = request.args.get("status", "All")

        page = max(1, request.args.get("page", 1, type=int) or 1)
        page_size = 20

        if selected_status != "All":
            data = data[
                data["Customer_Status"] == selected_status
            ]

        if search:
            searchable = data[
                ["Customer_Reference", "Country", "City"]
            ].fillna("").astype(str)

            matches = searchable.apply(
                lambda column: column.str.contains(
                    search,
                    case=False,
                    regex=False
                )
            ).any(axis=1)

            data = data[matches]

        total_customers = len(data)

        total_pages = max(
            1,
            (total_customers + page_size - 1) // page_size
        )

        page = min(page, total_pages)
        start = (page - 1) * page_size

        customers = data.iloc[
            start:start + page_size
        ].fillna("").to_dict(orient="records")

        return render_template(
            "customers.html",
            customers=customers,
            total_customers=total_customers,
            total_pages=total_pages,
            page=page,
            search=search,
            selected_status=selected_status,
            error=None
        )

    except Exception as error:
        return render_template(
            "customers.html",
            customers=[],
            total_customers=0,
            total_pages=1,
            page=1,
            search="",
            selected_status="All",
            error=str(error)
        )

@app.route("/customers/<reference>")
def customer_profile(reference):
    try:
        data = prepare_customer_data()

        if not reference.startswith("REC-") or not reference[4:].isdigit():
            return "Invalid customer reference", 404

        customer_number = int(reference[4:])

        if customer_number < 1 or customer_number > len(data):
            return "Customer not found", 404

        customer = data.iloc[
            customer_number - 1
        ].fillna("").to_dict()

        return render_template(
            "customer_profile.html",
            customer=customer
        )

    except Exception as error:
        return str(error), 500

@app.route("/retention")
def retention_page():
    try:
        data = prepare_retention_data()

        total_at_risk = int(
            (data["Engagement_Level"] != "Good").sum()
        )

        high_priority = int(
            (data["Priority_Level"] == "High").sum()
        )

        customers = data.head(100).fillna("").to_dict(
            orient="records"
        )

        return render_template(
            "retention.html",
            customers=customers,
            total_at_risk=total_at_risk,
            high_priority=high_priority,
            total_customers=len(data),
            error=None
        )

    except Exception as error:
        return render_template(
            "retention.html",
            customers=[],
            total_at_risk=0,
            high_priority=0,
            total_customers=0,
            error=str(error)
        )

@app.route("/retention/export")
def export_retention():
    try:
        data = prepare_retention_data()

        columns = [
            "Customer_Reference",
            "Country",
            "City",
            "Customer_Status",
            "Churn_Probability_Percentage",
            "Predicted_Churn",
            "Risk_Level",
            "Engagement_Level",
            "Engagement_Indicators",
            "Retention_Recommendations",
            "Priority_Score",
            "Priority_Level"
        ]

        output = BytesIO()

        data[columns].to_csv(output, index=False)
        output.seek(0)

        return send_file(
            output,
            mimetype="text/csv",
            as_attachment=True,
            download_name="reconnect_retention_report.csv"
        )

    except Exception as error:
        return str(error), 500

if __name__ == "__main__":
    app.run(debug=True)
