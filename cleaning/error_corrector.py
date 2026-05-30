import pandas as pd
import logging

logger = logging.getLogger(__name__)

class ErrorCorrector:
    """Detects and nulls out impossible/out-of-range values."""

    RULES = {
        "annual_income": lambda x: x < 0,
        "credit_score":  lambda x: (x < 300) | (x > 850),
        "age":           lambda x: x < 18,
    }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, rule in self.RULES.items():
            if col in df.columns:
                mask              = rule(df[col])
                count             = mask.sum()
                df.loc[mask, col] = None
                logger.info(f"[ErrorCorrector] {col}: nulled {count:,} invalid values")
                print(f"[ErrorCorrector] {col}: nulled {count:,} invalid values")
        return df