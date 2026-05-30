import pandas as pd
import numpy as np
import joblib
import os
import logging
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "model.pkl")
THRESHOLD_PATH= os.path.join(BASE_DIR, "..", "training", "artifacts", "threshold.pkl")
ENCODER_PATH  = os.path.join(BASE_DIR, "..", "feature_engineering", "artifacts", "encoder.pkl")
SCALER_PATH   = os.path.join(BASE_DIR, "..", "feature_engineering", "artifacts", "scaler.pkl")

app = Flask(__name__)

# ── Load artifacts ────────────────────────────────────────────────────────────
model     = joblib.load(MODEL_PATH)
threshold = joblib.load(THRESHOLD_PATH)
label_maps= joblib.load(ENCODER_PATH)

from sklearn.preprocessing import RobustScaler
scaler    = joblib.load(SCALER_PATH)

logger.info(f"Model loaded | Threshold: {threshold}")

# ── Constants ─────────────────────────────────────────────────────────────────
DROP_COLS = [
    "monthly_income", "months_employed", "monthly_payment",
    "employment_years", "num_dependents", "num_prev_loans",
    "credit_util_rate", "interest_rate", "num_credit_lines"
]

ONE_HOT_COLS = ["education_level", "employment_type"]
LABEL_COLS   = ["region", "marital_status", "property_type", "loan_purpose"]

SCALE_COLS   = [
    "annual_income", "savings_balance", "investment_balance",
    "loan_amount", "debt_to_income", "loan_to_income"
]

# ── Preprocessor ──────────────────────────────────────────────────────────────
class RequestPreprocessor:

    def _drop_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    def _add_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        df["debt_x_loan"]    = df["debt_to_income"] * df["loan_to_income"]
        df["income_per_loan"]= df["annual_income"]  / (df["loan_amount"] + 1e-9)
        df["score_x_dti"]    = df["credit_score"]   * df["debt_to_income"]
        df["savings_to_loan"]= df["savings_balance"] / (df["loan_amount"] + 1e-9)
        df["late_x_dti"]     = df["num_late_payments"] * df["debt_to_income"]
        df["age_x_income"]   = df["age"] * df["annual_income"]
        df["loan_burden"]    = df["loan_amount"] / (df["annual_income"] + df["savings_balance"] + 1e-9)
        return df

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ONE_HOT_COLS:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=False).astype(int)
                df      = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        for col in LABEL_COLS:
            if col in df.columns and col in label_maps:
                df[col] = df[col].map(label_maps[col]).fillna(-1).astype(int)
        return df

    def _scale(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in SCALE_COLS if c in df.columns]
        df[cols] = scaler.transform(df[cols])
        return df

    def _align_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        expected = model.feature_name_
        for col in expected:
            if col not in df.columns:
                df[col] = 0
        return df[expected]

    def transform(self, data: dict) -> pd.DataFrame:
        df = pd.DataFrame([data])
        df = self._drop_features(df)
        df = self._add_interactions(df)
        df = self._encode(df)
        df = self._scale(df)
        df = self._align_columns(df)
        return df

preprocessor = RequestPreprocessor()

# ── Risk tier ─────────────────────────────────────────────────────────────────
def get_risk_tier(prob: float) -> str:
    if prob >= 0.70:   return "High"
    elif prob >= 0.40: return "Medium"
    else:              return "Low"

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":    "healthy",
        "model":     "LightGBM",
        "threshold": threshold
    }), 200

@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "model_type":    "LightGBM",
        "threshold":     threshold,
        "n_estimators":  model.n_estimators_,
        "num_features":  len(model.feature_name_),
        "features":      model.feature_name_,
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        if not payload or "data" not in payload:
            return jsonify({"error": "Request body must contain a 'data' key"}), 400

        records  = payload["data"]
        if not isinstance(records, list):
            records = [records]

        results = []
        for record in records:
            df          = preprocessor.transform(record)
            prob        = float(model.predict_proba(df)[0][1])
            prediction  = int(prob >= threshold)
            results.append({
                "prediction":    prediction,
                "default_prob":  round(prob, 4),
                "risk_tier":     get_risk_tier(prob),
                "label":         "Default" if prediction == 1 else "No Default"
            })

        return jsonify({
            "results": results,
            "count":   len(results),
            "threshold": threshold
        }), 200

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting scoring API on port 8080 ...")
    app.run(host="0.0.0.0", port=8080, debug=False)