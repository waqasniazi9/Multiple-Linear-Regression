import json
from pathlib import Path
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from train_model import train, FEATURES, NUMERIC, CATEGORICAL, load_data

app = Flask(__name__, static_folder="static")

# Load model and data once at startup
print("Initializing House Price Predictor Model...")
model, report, comparison, data, X_train = train()
regressor = model.named_steps["regressor"]
preprocessor = model.named_steps["preprocess"]
scaler = preprocessor.named_transformers_["numeric"].named_steps["scale"]
encoder = preprocessor.named_transformers_["categorical"].named_steps["encode"]

# Precompute city statistics
city_stats = (
    data.groupby("city")
    .agg(
        avg_price=("price", "mean"),
        median_price=("price", "median"),
        count=("price", "count"),
        avg_sqft=("sqft_living", "mean"),
    )
    .reset_index()
    .sort_values(by="avg_price", ascending=False)
    .to_dict(orient="records")
)

AVAILABLE_CITIES = sorted(list(X_train["city"].unique()))


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory("static", path)


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    return jsonify(
        {
            "report": report,
            "total_cities": len(AVAILABLE_CITIES),
            "sample_size": len(data),
            "features_used": FEATURES,
        }
    )


@app.route("/api/cities", methods=["GET"])
def get_cities():
    return jsonify(
        {
            "cities": AVAILABLE_CITIES,
            "stats": city_stats,
        }
    )


@app.route("/api/sample-data", methods=["GET"])
def get_sample_data():
    sample = (
        data.sample(min(25, len(data)), random_state=42)
        .assign(price_formatted=lambda df: df["price"].apply(lambda p: f"${p:,.0f}"))
        .to_dict(orient="records")
    )
    return jsonify({"samples": sample})


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        req = request.get_json(force=True)

        city = str(req.get("city", "Seattle"))
        bedrooms = float(req.get("bedrooms", 3))
        bathrooms = float(req.get("bathrooms", 2.0))
        sqft_living = float(req.get("sqft_living", 1800))
        sqft_lot = float(req.get("sqft_lot", 6000))
        floors = float(req.get("floors", 1.0))
        waterfront = int(1 if req.get("waterfront") in [True, 1, "1", "true"] else 0)
        view = int(req.get("view", 0))
        condition = int(req.get("condition", 3))
        sqft_basement = float(req.get("sqft_basement", 0))
        yr_built = int(req.get("yr_built", 1990))
        yr_renovated = int(req.get("yr_renovated", 0))

        # Validation rules
        validation_warnings = []
        if sqft_basement > sqft_living:
            return jsonify({"error": "Basement square footage cannot exceed total living square footage."}), 400
        if yr_renovated > 0 and yr_renovated < yr_built:
            return jsonify({"error": "Renovation year cannot precede construction year."}), 400
        if sqft_living < 100 or sqft_living > 40000:
            validation_warnings.append("Living area is outside standard residential range.")

        record = {
            "city": city,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "sqft_living": sqft_living,
            "sqft_lot": sqft_lot,
            "floors": floors,
            "sqft_basement": sqft_basement,
            "waterfront": waterfront,
            "condition": condition,
            "view": view,
            "yr_built": yr_built,
            "yr_renovated": yr_renovated,
        }

        row = pd.DataFrame([record])[FEATURES]
        pred_val = float(model.predict(row)[0])

        mae = float(report["model"]["MAE"])
        lower_bound = max(50000.0, pred_val - mae)
        upper_bound = pred_val + mae

        price_per_sqft = (pred_val / sqft_living) if sqft_living > 0 else 0

        # Financial estimate calculations (Mortgage: 30-year fixed @ 6.5%, 20% down payment)
        down_payment = pred_val * 0.20
        loan_amount = pred_val * 0.80
        monthly_rate = 0.065 / 12
        num_payments = 30 * 12
        if loan_amount > 0:
            monthly_mortgage = (
                loan_amount
                * (monthly_rate * (1 + monthly_rate) ** num_payments)
                / ((1 + monthly_rate) ** num_payments - 1)
            )
        else:
            monthly_mortgage = 0

        # Feature impact breakdown (approximate linear components)
        transformed = preprocessor.transform(row)
        cat_feature_names = list(encoder.get_feature_names_out(CATEGORICAL))
        feature_names = NUMERIC + cat_feature_names
        coefs = regressor.coef_

        contributions = []
        for name, coef, val in zip(feature_names, coefs, transformed[0]):
            contribution = float(coef * val)
            clean_name = name.replace("city_", "City: ").replace("_", " ").title()
            contributions.append({
                "feature": clean_name,
                "raw_feature": name,
                "impact": contribution,
            })

        # Sort by absolute impact
        contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)
        top_positive = [c for c in contributions if c["impact"] > 0][:5]
        top_negative = [c for c in contributions if c["impact"] < 0][:5]

        # City benchmark comparison
        city_match = next((item for item in city_stats if item["city"].lower() == city.lower()), None)
        city_benchmark = city_match["avg_price"] if city_match else None

        return jsonify({
            "success": True,
            "predicted_price": round(pred_val, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
            "price_per_sqft": round(price_per_sqft, 2),
            "mortgage_estimate": {
                "down_payment_20": round(down_payment, 2),
                "monthly_payment": round(monthly_mortgage, 2),
                "assumed_rate": "6.50% APR",
                "loan_term": "30-Year Fixed",
            },
            "city_benchmark": city_benchmark,
            "benchmark_diff": round(pred_val - city_benchmark, 2) if city_benchmark else 0,
            "top_drivers": contributions[:7],
            "warnings": validation_warnings,
            "inputs": record,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting Flask House Price Predictor Server on http://localhost:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
