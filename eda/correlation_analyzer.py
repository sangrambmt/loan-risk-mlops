import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

logger = logging.getLogger(__name__)

PLOTS_DIR   = os.path.join(os.path.dirname(__file__), "plots")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")

class CorrelationAnalyzer:
    """Computes correlation matrix and plots a heatmap."""

    NUMERIC_COLS = [
        "age", "employment_years", "annual_income", "monthly_income",
        "savings_balance", "investment_balance", "credit_score",
        "num_credit_lines", "num_late_payments", "credit_util_rate",
        "loan_amount", "loan_term_months", "interest_rate",
        "monthly_payment", "debt_to_income", "loan_to_income",
        "num_dependents", "num_prev_loans", "has_cosigner", "default"
    ]

    def __init__(self):
        os.makedirs(PLOTS_DIR,   exist_ok=True)
        os.makedirs(REPORTS_DIR, exist_ok=True)

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        cols    = [c for c in self.NUMERIC_COLS if c in df.columns]
        corr    = df[cols].corr().round(4)

        # Print top correlations with target
        if "default" in corr.columns:
            top = (corr["default"]
                   .drop("default")
                   .abs()
                   .sort_values(ascending=False))
            print(f"\n{'='*50}")
            print(f"  TOP CORRELATIONS WITH TARGET (default)")
            print(f"{'='*50}")
            print(top.to_string())
            print(f"{'='*50}\n")

        # Save report
        out_csv = os.path.join(REPORTS_DIR, "correlation_report.csv")
        corr.to_csv(out_csv)
        logger.info(f"[CorrelationAnalyzer] Report saved -> {out_csv}")
        print(f"[CorrelationAnalyzer] Report saved -> {out_csv}")

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(18, 14))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    linewidths=0.5, annot_kws={"size": 7},
                    square=True, ax=ax)
        ax.set_title("Correlation Matrix", fontsize=14)
        plt.tight_layout()
        out_png = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
        plt.savefig(out_png, bbox_inches="tight", dpi=150)
        plt.close()
        logger.info(f"[CorrelationAnalyzer] Heatmap saved -> {out_png}")
        print(f"[CorrelationAnalyzer] Heatmap saved -> {out_png}")

        return corr