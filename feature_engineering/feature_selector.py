import pandas as pd
import logging

logger = logging.getLogger(__name__)

class FeatureSelector:
    """Drops high-correlation and low-signal features based on EDA findings."""

    HIGH_CORRELATION = {
        "monthly_income":  "98.4% correlated with annual_income",
        "months_employed": "derived directly from employment_years",
        "monthly_payment": "78.4% correlated with debt_to_income",
    }

    LOW_SIGNAL = {
        "employment_years":  "near-zero correlation with target (0.001)",
        "num_dependents":    "near-zero correlation with target (0.0002)",
        "num_prev_loans":    "near-zero correlation with target (-0.003)",
        "credit_util_rate":  "near-zero correlation with target (-0.0001)",
        "interest_rate":     "near-zero correlation with target (0.006)",
        "num_credit_lines":  "near-zero correlation with target (0.005)",
    }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"\n{'='*55}")
        print(f"  FEATURE SELECTION")
        print(f"{'='*55}")

        # Drop high correlation
        for col, reason in self.HIGH_CORRELATION.items():
            if col in df.columns:
                df = df.drop(columns=[col])
                logger.info(f"[FeatureSelector] Dropped '{col}' — {reason}")
                print(f"  [HIGH CORR ] Dropped '{col}' — {reason}")

        # Drop low signal
        for col, reason in self.LOW_SIGNAL.items():
            if col in df.columns:
                df = df.drop(columns=[col])
                logger.info(f"[FeatureSelector] Dropped '{col}' — {reason}")
                print(f"  [LOW SIGNAL] Dropped '{col}' — {reason}")

        print(f"\n  Remaining columns : {df.shape[1]}")
        print(f"  Remaining rows    : {df.shape[0]:,}")
        print(f"{'='*55}\n")
        return df