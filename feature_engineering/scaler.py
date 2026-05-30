import pandas as pd
import logging
import joblib
import os
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

class Scaler:
    """Applies RobustScaler to high-outlier numeric columns."""

    SCALE_COLS = [
        "annual_income",
        "savings_balance",
        "investment_balance",
        "loan_amount",
        "debt_to_income",
        "loan_to_income",
    ]

    def __init__(self):
        self._scaler = RobustScaler()
        self._cols_to_scale: list = []
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    def fit(self, df: pd.DataFrame) -> "Scaler":
        self._cols_to_scale = [c for c in self.SCALE_COLS if c in df.columns]
        self._scaler.fit(df[self._cols_to_scale])
        logger.info(f"[Scaler] Fitted on columns: {self._cols_to_scale}")
        print(f"[Scaler] Fitted on columns: {self._cols_to_scale}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self._cols_to_scale] = self._scaler.transform(df[self._cols_to_scale])
        logger.info(f"[Scaler] Scaled {len(self._cols_to_scale)} columns")
        print(f"[Scaler] Scaled {len(self._cols_to_scale)} columns")
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def save(self):
        path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
        joblib.dump(self._scaler, path)
        logger.info(f"[Scaler] Saved -> {path}")
        print(f"[Scaler] Saved -> {path}")

    def load(self):
        path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
        self._scaler = joblib.load(path)
        logger.info(f"[Scaler] Loaded -> {path}")
        print(f"[Scaler] Loaded -> {path}")
        return self