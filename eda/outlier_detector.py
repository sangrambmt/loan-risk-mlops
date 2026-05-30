import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")

class OutlierDetector:
    """Detects outliers using IQR method and generates a report."""

    NUMERIC_COLS = [
        "age", "employment_years", "annual_income", "monthly_income",
        "savings_balance", "investment_balance", "credit_score",
        "num_credit_lines", "num_late_payments", "credit_util_rate",
        "loan_amount", "loan_term_months", "interest_rate",
        "monthly_payment", "debt_to_income", "loan_to_income",
        "num_dependents", "num_prev_loans"
    ]

    def __init__(self, iqr_multiplier: float = 1.5):
        self.iqr_multiplier = iqr_multiplier
        os.makedirs(REPORTS_DIR, exist_ok=True)

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        cols    = [c for c in self.NUMERIC_COLS if c in df.columns]
        results = []

        for col in cols:
            q1  = df[col].quantile(0.25)
            q3  = df[col].quantile(0.75)
            iqr = q3 - q1
            lb  = q1 - self.iqr_multiplier * iqr
            ub  = q3 + self.iqr_multiplier * iqr

            outliers       = df[(df[col] < lb) | (df[col] > ub)]
            outlier_count  = len(outliers)
            outlier_pct    = round(outlier_count / len(df) * 100, 4)

            results.append({
                "column":        col,
                "q1":            round(q1,  4),
                "q3":            round(q3,  4),
                "iqr":           round(iqr, 4),
                "lower_bound":   round(lb,  4),
                "upper_bound":   round(ub,  4),
                "outlier_count": outlier_count,
                "outlier_pct":   outlier_pct,
            })

        report = pd.DataFrame(results).sort_values("outlier_count", ascending=False)

        # Print
        print(f"\n{'='*60}")
        print(f"  OUTLIER DETECTION REPORT  (IQR x {self.iqr_multiplier})")
        print(f"{'='*60}")
        print(report.to_string(index=False))
        print(f"{'='*60}\n")

        # Save
        out = os.path.join(REPORTS_DIR, "outlier_report.csv")
        report.to_csv(out, index=False)
        logger.info(f"[OutlierDetector] Report saved -> {out}")
        print(f"[OutlierDetector] Report saved -> {out}")

        return report