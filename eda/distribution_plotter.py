import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

logger = logging.getLogger(__name__)

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")

class DistributionPlotter:
    """Plots histograms and boxplots for all numeric columns."""

    NUMERIC_COLS = [
        "age", "employment_years", "annual_income", "monthly_income",
        "savings_balance", "investment_balance", "credit_score",
        "num_credit_lines", "num_late_payments", "credit_util_rate",
        "loan_amount", "loan_term_months", "interest_rate",
        "monthly_payment", "debt_to_income", "loan_to_income",
        "num_dependents", "num_prev_loans"
    ]

    def __init__(self):
        os.makedirs(PLOTS_DIR, exist_ok=True)

    def plot_histograms(self, df: pd.DataFrame):
        cols = [c for c in self.NUMERIC_COLS if c in df.columns]
        n    = len(cols)
        fig, axes = plt.subplots(nrows=(n + 3) // 4, ncols=4, figsize=(20, (n + 3) // 4 * 4))
        axes = axes.flatten()

        for i, col in enumerate(cols):
            axes[i].hist(df[col].dropna(), bins=50, color="steelblue", edgecolor="white")
            axes[i].set_title(col, fontsize=10)
            axes[i].set_xlabel("")
            axes[i].set_ylabel("Frequency")

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle("Feature Distributions", fontsize=14, y=1.01)
        plt.tight_layout()
        out = os.path.join(PLOTS_DIR, "histograms.png")
        plt.savefig(out, bbox_inches="tight", dpi=150)
        plt.close()
        logger.info(f"[DistributionPlotter] Histograms saved -> {out}")
        print(f"[DistributionPlotter] Histograms saved -> {out}")

    def plot_boxplots(self, df: pd.DataFrame):
        cols = [c for c in self.NUMERIC_COLS if c in df.columns]
        n    = len(cols)
        fig, axes = plt.subplots(nrows=(n + 3) // 4, ncols=4, figsize=(20, (n + 3) // 4 * 4))
        axes = axes.flatten()

        for i, col in enumerate(cols):
            axes[i].boxplot(df[col].dropna(), patch_artist=True,
                            boxprops=dict(facecolor="steelblue", color="navy"),
                            medianprops=dict(color="red", linewidth=2))
            axes[i].set_title(col, fontsize=10)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle("Feature Boxplots", fontsize=14, y=1.01)
        plt.tight_layout()
        out = os.path.join(PLOTS_DIR, "boxplots.png")
        plt.savefig(out, bbox_inches="tight", dpi=150)
        plt.close()
        logger.info(f"[DistributionPlotter] Boxplots saved -> {out}")
        print(f"[DistributionPlotter] Boxplots saved -> {out}")

    def plot(self, df: pd.DataFrame):
        self.plot_histograms(df)
        self.plot_boxplots(df)