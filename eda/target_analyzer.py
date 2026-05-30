import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

logger = logging.getLogger(__name__)

PLOTS_DIR   = os.path.join(os.path.dirname(__file__), "plots")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")

class TargetAnalyzer:
    """Analyzes default rate across all features against the target column."""

    CATEGORICAL_COLS = [
        "education_level", "marital_status", "region",
        "employment_type", "loan_purpose", "property_type"
    ]

    NUMERIC_COLS = [
        "age", "credit_score", "annual_income", "debt_to_income",
        "loan_amount", "interest_rate", "num_late_payments",
        "employment_years", "credit_util_rate", "loan_to_income"
    ]

    def __init__(self, target_col: str = "default"):
        self.target_col = target_col
        os.makedirs(PLOTS_DIR,   exist_ok=True)
        os.makedirs(REPORTS_DIR, exist_ok=True)

    def _plot_categorical(self, df: pd.DataFrame):
        cols = [c for c in self.CATEGORICAL_COLS if c in df.columns]
        fig, axes = plt.subplots(nrows=(len(cols) + 1) // 2, ncols=2,
                                 figsize=(16, len(cols) * 3))
        axes = axes.flatten()

        for i, col in enumerate(cols):
            rate = df.groupby(col)[self.target_col].mean().sort_values(ascending=False)
            axes[i].bar(rate.index, rate.values, color="steelblue", edgecolor="white")
            axes[i].set_title(f"Default Rate by {col}", fontsize=11)
            axes[i].set_ylabel("Default Rate")
            axes[i].set_xlabel(col)
            axes[i].tick_params(axis="x", rotation=30)
            for j, v in enumerate(rate.values):
                axes[i].text(j, v + 0.002, f"{v:.2%}", ha="center", fontsize=8)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle("Default Rate by Categorical Features", fontsize=14, y=1.01)
        plt.tight_layout()
        out = os.path.join(PLOTS_DIR, "target_categorical.png")
        plt.savefig(out, bbox_inches="tight", dpi=150)
        plt.close()
        logger.info(f"[TargetAnalyzer] Categorical plot saved -> {out}")
        print(f"[TargetAnalyzer] Categorical plot saved -> {out}")

    def _plot_numeric(self, df: pd.DataFrame):
        cols = [c for c in self.NUMERIC_COLS if c in df.columns]
        fig, axes = plt.subplots(nrows=(len(cols) + 1) // 2, ncols=2,
                                 figsize=(16, len(cols) * 3))
        axes = axes.flatten()

        for i, col in enumerate(cols):
            for val, color, label in [(0, "steelblue", "No Default"),
                                      (1, "tomato",    "Default")]:
                subset = df[df[self.target_col] == val][col].dropna()
                axes[i].hist(subset, bins=40, alpha=0.6, color=color,
                             label=label, edgecolor="white")
            axes[i].set_title(f"{col} by Default", fontsize=11)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Frequency")
            axes[i].legend()

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle("Numeric Feature Distributions by Default", fontsize=14, y=1.01)
        plt.tight_layout()
        out = os.path.join(PLOTS_DIR, "target_numeric.png")
        plt.savefig(out, bbox_inches="tight", dpi=150)
        plt.close()
        logger.info(f"[TargetAnalyzer] Numeric plot saved -> {out}")
        print(f"[TargetAnalyzer] Numeric plot saved -> {out}")

    def _save_report(self, df: pd.DataFrame):
        results = []
        for col in self.CATEGORICAL_COLS:
            if col in df.columns:
                rate = df.groupby(col)[self.target_col].mean().reset_index()
                rate.columns = ["value", "default_rate"]
                rate.insert(0, "column", col)
                results.append(rate)
        report = pd.concat(results, ignore_index=True).round(4)
        out    = os.path.join(REPORTS_DIR, "target_report.csv")
        report.to_csv(out, index=False)
        logger.info(f"[TargetAnalyzer] Report saved -> {out}")
        print(f"[TargetAnalyzer] Report saved -> {out}")

        overall = df[self.target_col].mean()
        print(f"\n{'='*50}")
        print(f"  TARGET SUMMARY")
        print(f"{'='*50}")
        print(f"  Overall Default Rate : {overall:.2%}")
        print(f"  Default  (1)         : {df[self.target_col].sum():,}")
        print(f"  No Default (0)       : {(df[self.target_col] == 0).sum():,}")
        print(f"{'='*50}\n")

    def analyze(self, df: pd.DataFrame):
        self._save_report(df)
        self._plot_categorical(df)
        self._plot_numeric(df)