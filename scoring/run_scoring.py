import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "..", "data")
MODEL_PATH     = os.path.join(BASE_DIR, "model.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "..", "training", "artifacts", "threshold.pkl")
ENCODER_PATH   = os.path.join(BASE_DIR, "..", "feature_engineering", "artifacts", "encoder.pkl")
SCALER_PATH    = os.path.join(BASE_DIR, "..", "feature_engineering", "artifacts", "scaler.pkl")

# ── Constants ─────────────────────────────────────────────────────────────────
DROP_COLS    = [
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
class BatchPreprocessor:
    """Applies full feature engineering pipeline to scoring data."""

    def __init__(self, label_maps: dict, scaler, model):
        self.label_maps = label_maps
        self.scaler     = scaler
        self.model      = model

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Drop
        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

        # Interactions
        df["debt_x_loan"]     = df["debt_to_income"] * df["loan_to_income"]
        df["income_per_loan"] = df["annual_income"]  / (df["loan_amount"] + 1e-9)
        df["score_x_dti"]     = df["credit_score"]   * df["debt_to_income"]
        df["savings_to_loan"] = df["savings_balance"] / (df["loan_amount"] + 1e-9)
        df["late_x_dti"]      = df["num_late_payments"] * df["debt_to_income"]
        df["age_x_income"]    = df["age"] * df["annual_income"]
        df["loan_burden"]     = df["loan_amount"] / (df["annual_income"] + df["savings_balance"] + 1e-9)

        # One-hot encode
        for col in ONE_HOT_COLS:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=False).astype(int)
                df      = pd.concat([df.drop(columns=[col]), dummies], axis=1)

        # Label encode
        for col in LABEL_COLS:
            if col in df.columns and col in self.label_maps:
                df[col] = df[col].map(self.label_maps[col]).fillna(-1).astype(int)

        # Scale
        cols = [c for c in SCALE_COLS if c in df.columns]
        df[cols] = self.scaler.transform(df[cols])

        # Align columns
        expected = self.model.feature_name_
        for col in expected:
            if col not in df.columns:
                df[col] = 0
        return df[expected]

# ── Scorer ────────────────────────────────────────────────────────────────────
class BatchScorer:
    """Scores data in batches and assigns risk tiers."""

    def __init__(self, batch_size: int = 5000):
        self.batch_size = batch_size
        self.model      = joblib.load(MODEL_PATH)
        self.threshold  = joblib.load(THRESHOLD_PATH)
        self.label_maps = joblib.load(ENCODER_PATH)
        self.scaler     = joblib.load(SCALER_PATH)
        self.preprocessor = BatchPreprocessor(self.label_maps, self.scaler, self.model)
        logger.info(f"[BatchScorer] Model loaded | Threshold: {self.threshold}")

    def _get_risk_tier(self, prob: float) -> str:
        if prob >= 0.70:   return "High"
        elif prob >= 0.40: return "Medium"
        else:              return "Low"

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared  = self.preprocessor.transform(df)
        n_batches = (len(prepared) + self.batch_size - 1) // self.batch_size
        all_probs = []
        all_preds = []

        logger.info(f"[BatchScorer] Scoring {len(df):,} rows in {n_batches} batches ...")
        for i in range(n_batches):
            batch      = prepared.iloc[i * self.batch_size:(i + 1) * self.batch_size]
            probs      = self.model.predict_proba(batch)[:, 1]
            preds      = (probs >= self.threshold).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            print(f"  Batch {i+1}/{n_batches} scored — {len(batch):,} rows")

        df = df.copy()
        df["default_prob"] = np.round(all_probs, 4)
        df["prediction"]   = all_preds
        df["risk_tier"]    = [self._get_risk_tier(p) for p in all_probs]
        df["label"]        = df["prediction"].map({0: "No Default", 1: "Default"})
        df["scored_at"]    = datetime.utcnow().isoformat()
        return df

# ── Pipeline ──────────────────────────────────────────────────────────────────
class ScoringPipeline:
    """Orchestrates batch scoring pipeline."""

    def __init__(self):
        self.scorer = BatchScorer()

    def _print_report(self, df: pd.DataFrame):
        print(f"\n{'='*55}")
        print(f"  SCORING SUMMARY REPORT")
        print(f"{'='*55}")
        print(f"  Scored at        : {datetime.utcnow().isoformat()}")
        print(f"  Total records    : {len(df):,}")
        print(f"  Default rate     : {df['prediction'].mean():.2%}")
        print(f"  Avg default prob : {df['default_prob'].mean():.4f}")
        print(f"{'='*55}")
        print(f"  🔴 High risk     : {(df['risk_tier'] == 'High').sum():,}")
        print(f"  🟡 Medium risk   : {(df['risk_tier'] == 'Medium').sum():,}")
        print(f"  🟢 Low risk      : {(df['risk_tier'] == 'Low').sum():,}")
        print(f"{'='*55}\n")

    def run(self):
        logger.info("=== Batch Scoring Pipeline Started ===")

        # Load
        path = os.path.join(DATA_DIR, "scoring_data.csv")
        df   = pd.read_csv(path)
        logger.info(f"[Pipeline] Loaded scoring data: {df.shape}")

        # Score
        scored = self.scorer.score(df)

        # Save
        out = os.path.join(DATA_DIR, "scored_output.csv")
        scored.to_csv(out, index=False)
        logger.info(f"[Pipeline] Scored output saved -> {out}")

        # Report
        self._print_report(scored)
        logger.info("=== Batch Scoring Pipeline Complete ===")

if __name__ == "__main__":
    ScoringPipeline().run()