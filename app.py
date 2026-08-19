from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import os

app = Flask(__name__)

# -----------------------------
# Load and prepare the dataset
# -----------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "data.csv")

df = pd.read_csv(DATA_PATH)

# Same preprocessing logic as the uploaded training code:
# date -> month/day/year, then remove date.
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["year"] = df["date"].dt.year
df.drop("date", axis=1, inplace=True)

# Store the mappings so the web form uses the same encoding style.
city_values = sorted(df["city"].dropna().astype(str).unique())
country_values = sorted(df["country"].dropna().astype(str).unique())

city_map = {value: i for i, value in enumerate(city_values)}
country_map = {value: i for i, value in enumerate(country_values)}

df["city"] = df["city"].astype(str).map(city_map)
df["country"] = df["country"].astype(str).map(country_map)

# Uploaded training code uses price as Y and the remaining columns as X.
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

test_predictions = model.predict(X_test)
r2 = r2_score(y_test, test_predictions)
mse = mean_squared_error(y_test, test_predictions)
rmse = float(np.sqrt(mse))

# Numeric input columns used by the model.
numeric_fields = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "month",
    "day",
    "year",
]


@app.route("/")
def home():
    return render_template(
        "index.html",
        cities=city_values,
        countries=country_values,
        r2=round(r2, 4),
        rmse=round(rmse, 2),
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or request.form

        # Build one row in exactly the same feature order used during training.
        row = {
            "bedrooms": float(data["bedrooms"]),
            "bathrooms": float(data["bathrooms"]),
            "sqft_living": float(data["sqft_living"]),
            "sqft_lot": float(data["sqft_lot"]),
            "floors": float(data["floors"]),
            "waterfront": float(data["waterfront"]),
            "view": float(data["view"]),
            "condition": float(data["condition"]),
            "sqft_above": float(data["sqft_above"]),
            "sqft_basement": float(data["sqft_basement"]),
            "yr_built": float(data["yr_built"]),
            "yr_renovated": float(data["yr_renovated"]),
            "city": float(city_map.get(str(data["city"]), 0)),
            "country": float(country_map.get(str(data["country"]), 0)),
            "month": float(data["month"]),
            "day": float(data["day"]),
            "year": float(data["year"]),
        }

        # X contains these columns in this exact order after preprocessing.
        feature_order = list(X.columns)
        input_df = pd.DataFrame([row])[feature_order]

        prediction = float(model.predict(input_df)[0])

        return jsonify({
            "success": True,
            "prediction": round(prediction, 2),
            "formatted_prediction": f"${prediction:,.2f}",
            "model_r2": round(r2, 4),
            "model_rmse": round(rmse, 2),
        })

    except Exception as exc:
        return jsonify({
            "success": False,
            "error": str(exc)
        }), 400


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": "LinearRegression",
        "dataset_rows": int(len(df))
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
