import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")

class StatisticsAnalyzer:
    """Computes descriptive statistics, skewness and kurtosis for numeric columns."""

    NUMERIC_COLS = [
        "age", "employment_years", "months_employed", "annual_income",
        "monthly_income", "savings_balance", "investment_balance",
        "credit_score", "num_credit_lines", "num_late_payments",
        "credit_util_rate", "loan_amount", "loan_term_months",
        "interest_rate", "monthly_payment", "debt_to_income",
        "loan_to_income", "num_dependents", "has_cosigner", "num_prev_loans"
    ]

    def __init__(self):
        os.makedirs(REPORTS_DIR, exist_ok=True)

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        cols     = [c for c in self.NUMERIC_COLS if c in df.columns]
        subset   = df[cols]

        stats    = subset.describe().T
        stats["skewness"] = subset.skew()
        stats["kurtosis"] = subset.kurt()
        stats["missing"]  = df[cols].isnull().sum()
        stats["missing%"] = (df[cols].isnull().mean() * 100).round(2)

        stats = stats[["count", "mean", "std", "min", "25%", "50%", "75%", "max",
                        "skewness", "kurtosis", "missing", "missing%"]]
        stats = stats.round(4)

        # Print
        print(f"\n{'='*60}")
        print(f"  STATISTICS REPORT")
        print(f"{'='*60}")
        print(stats.to_string())
        print(f"{'='*60}\n")

        # Save
        out = os.path.join(REPORTS_DIR, "statistics_report.csv")
        stats.to_csv(out)
        logger.info(f"[StatisticsAnalyzer] Report saved -> {out}")
        print(f"[StatisticsAnalyzer] Report saved -> {out}")

        return stats