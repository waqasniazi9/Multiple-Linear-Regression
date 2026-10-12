from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

NUMERIC = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view", "condition", "sqft_basement", "yr_built", "yr_renovated"]
CATEGORICAL = ["city"]
FEATURES = NUMERIC + CATEGORICAL

def load_data():
    csv_path = Path(__file__).parent / "data" / "data.csv"
    if not csv_path.exists():
        csv_path = Path(__file__).parent / "data.csv"
    raw = pd.read_csv(csv_path)
    clean = raw.loc[raw.price > 0].drop_duplicates().copy()
    return raw, clean

def train():
    raw, data = load_data()
    X_train, X_test, y_train, y_test = train_test_split(data[FEATURES], data.price, test_size=0.2, random_state=42)
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(drop="first", handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)], sparse_threshold=0)
    model = Pipeline([("preprocess", preprocessor), ("regressor", LinearRegression())])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train).predict(X_test)
    def metrics(y, p):
        return {"MAE": float(mean_absolute_error(y,p)), "RMSE": float(np.sqrt(mean_squared_error(y,p))), "R2": float(r2_score(y,p))}
    report = {"model": metrics(y_test, predictions), "baseline": metrics(y_test, baseline), "raw_rows": len(raw), "usable_rows": len(data), "train_rows": len(X_train), "test_rows": len(X_test), "removed_nonpositive": int((raw.price <= 0).sum())}
    comparison = pd.DataFrame({"Actual": y_test, "Predicted": predictions, "City": X_test.city})
    comparison["Residual"] = comparison.Actual - comparison.Predicted
    return model, report, comparison, data, X_train

if __name__ == "__main__":
    import json
    import joblib
    model, report, comparison, data, X_train = train()
    output = Path(__file__).parent / "artifacts"
    output.mkdir(exist_ok=True)
    joblib.dump(model, output / "pipeline.joblib")
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    comparison.to_csv(output / "test_predictions.csv", index=False)
    print(json.dumps(report, indent=2))
