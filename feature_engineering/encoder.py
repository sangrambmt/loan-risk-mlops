import pandas as pd
import logging
import joblib
import os

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

class Encoder:
    """One-hot encodes high-signal categoricals, label encodes low-signal ones."""

    ONE_HOT_COLS = [
        "education_level",
        "employment_type",
    ]

    LABEL_COLS = [
        "region",
        "marital_status",
        "property_type",
        "loan_purpose",
    ]

    def __init__(self):
        self._label_maps: dict = {}
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    def fit(self, df: pd.DataFrame) -> "Encoder":
        for col in self.LABEL_COLS:
            if col in df.columns:
                categories         = sorted(df[col].dropna().unique())
                self._label_maps[col] = {cat: i for i, cat in enumerate(categories)}
        logger.info(f"[Encoder] Fitted label maps for: {list(self._label_maps.keys())}")
        print(f"[Encoder] Fitted label maps for: {list(self._label_maps.keys())}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # One-hot encode
        for col in self.ONE_HOT_COLS:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=False).astype(int)
                df      = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                logger.info(f"[Encoder] One-hot encoded '{col}' -> {dummies.shape[1]} columns")
                print(f"[Encoder] One-hot encoded '{col}' -> {dummies.shape[1]} columns")

        # Label encode
        for col, mapping in self._label_maps.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(-1).astype(int)
                logger.info(f"[Encoder] Label encoded '{col}' -> {mapping}")
                print(f"[Encoder] Label encoded '{col}' -> {mapping}")

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def save(self):
        path = os.path.join(ARTIFACTS_DIR, "encoder.pkl")
        joblib.dump(self._label_maps, path)
        logger.info(f"[Encoder] Saved -> {path}")
        print(f"[Encoder] Saved -> {path}")

    def load(self):
        path = os.path.join(ARTIFACTS_DIR, "encoder.pkl")
        self._label_maps = joblib.load(path)
        logger.info(f"[Encoder] Loaded -> {path}")
        print(f"[Encoder] Loaded -> {path}")
        return self