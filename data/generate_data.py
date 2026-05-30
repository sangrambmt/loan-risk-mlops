import pandas as pd
import numpy as np
import os, logging, warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Config ────────────────────────────────────────────────────────────────────
from dataclasses import dataclass

@dataclass
class DataConfig:
    total_rows:   int   = 100_000
    scoring_rows: int   = 10_000
    seed:         int   = 42

# ── Base Generator ────────────────────────────────────────────────────────────
class BaseGenerator:
    def __init__(self, config: DataConfig):
        self.config = config
        self.rng    = np.random.default_rng(config.seed)

    def _introduce_missing(self, df: pd.DataFrame, rate: float = 0.03) -> pd.DataFrame:
        nullable = ["annual_income", "credit_score", "employment_years",
                    "debt_to_income", "num_credit_lines", "months_employed",
                    "savings_balance", "investment_balance"]
        for col in nullable:
            if col in df.columns:
                mask = self.rng.random(len(df)) < rate
                df.loc[mask, col] = np.nan
        return df

    def _introduce_duplicates(self, df: pd.DataFrame, rate: float = 0.005) -> pd.DataFrame:
        n    = int(len(df) * rate)
        dups = df.sample(n=n, random_state=self.config.seed)
        return pd.concat([df, dups], ignore_index=True)

    def _introduce_errors(self, df: pd.DataFrame) -> pd.DataFrame:
        idx  = self.rng.integers(0, len(df), size=50)
        df.loc[idx, "annual_income"] = df.loc[idx, "annual_income"] * -1   # negative income
        idx2 = self.rng.integers(0, len(df), size=50)
        df.loc[idx2, "credit_score"] = 999                                  # impossible score
        idx3 = self.rng.integers(0, len(df), size=50)
        df.loc[idx3, "age"] = self.rng.integers(0, 10, size=50)            # implausible age
        return df

# ── Feature Builder ───────────────────────────────────────────────────────────
class FeatureBuilder(BaseGenerator):

    EMPLOYMENT_TYPES = ["Employed", "Self-Employed", "Part-Time", "Unemployed", "Retired"]
    EDUCATION_LEVELS = ["High School", "Associate", "Bachelor", "Master", "PhD"]
    LOAN_PURPOSES    = ["Home", "Car", "Education", "Medical", "Business", "Personal", "Vacation"]
    MARITAL_STATUSES = ["Single", "Married", "Divorced", "Widowed"]
    REGIONS          = ["North", "South", "East", "West", "Central"]
    PROPERTY_TYPES   = ["Rent", "Own", "Mortgage", "Other"]

    def build(self, n: int) -> pd.DataFrame:
        logger.info(f"Building {n:,} feature rows ...")

        # ── Demographics ──────────────────────────────────────────────────────
        age               = self.rng.integers(18, 75, size=n)
        education         = self.rng.choice(self.EDUCATION_LEVELS,  size=n, p=[0.20, 0.15, 0.35, 0.20, 0.10])
        marital_status    = self.rng.choice(self.MARITAL_STATUSES,  size=n, p=[0.30, 0.50, 0.15, 0.05])
        num_dependents    = self.rng.integers(0, 6, size=n)
        region            = self.rng.choice(self.REGIONS, size=n)

        # ── Employment ────────────────────────────────────────────────────────
        employment_type   = self.rng.choice(self.EMPLOYMENT_TYPES,  size=n, p=[0.55, 0.15, 0.10, 0.10, 0.10])
        employment_years  = np.clip(self.rng.normal(8, 5, n), 0, 45)
        months_employed   = (employment_years * 12).astype(int)

        # ── Financials ────────────────────────────────────────────────────────
        edu_multiplier    = np.select(
            [education == e for e in self.EDUCATION_LEVELS],
            [0.70, 0.85, 1.00, 1.30, 1.60]
        )
        base_income       = 30_000 + age * 800 + self.rng.normal(0, 8_000, n)
        annual_income     = np.clip(base_income * edu_multiplier, 10_000, 500_000)
        monthly_income    = annual_income / 12
        savings_balance   = np.clip(self.rng.exponential(annual_income * 0.25, n), 0, 1_000_000)
        investment_balance= np.clip(self.rng.exponential(annual_income * 0.15, n), 0, 800_000)

        # ── Credit ────────────────────────────────────────────────────────────
        credit_score      = np.clip(self.rng.normal(680, 80, n), 300, 850).astype(int)
        num_credit_lines  = self.rng.integers(0, 20, size=n)
        num_late_payments = self.rng.integers(0, 10, size=n)
        credit_util_rate  = np.clip(self.rng.beta(2, 5, n), 0, 1)

        # ── Loan ─────────────────────────────────────────────────────────────
        loan_purpose      = self.rng.choice(self.LOAN_PURPOSES, size=n)
        loan_amount       = np.clip(self.rng.lognormal(10, 1, n), 1_000, 500_000)
        loan_term_months  = self.rng.choice([12, 24, 36, 48, 60, 84, 120], size=n)
        interest_rate     = np.clip(self.rng.normal(7.5, 3, n), 1.5, 28.0)
        monthly_payment   = (loan_amount * (interest_rate / 100 / 12)) / \
                            (1 - (1 + interest_rate / 100 / 12) ** (-loan_term_months))

        # ── Derived Ratios ────────────────────────────────────────────────────
        debt_to_income    = np.clip(monthly_payment / (monthly_income + 1e-9), 0, 1)
        loan_to_income    = np.clip(loan_amount     / (annual_income  + 1e-9), 0, 10)
        property_type     = self.rng.choice(self.PROPERTY_TYPES, size=n, p=[0.35, 0.20, 0.35, 0.10])
        has_cosigner      = (self.rng.random(n) < 0.20).astype(int)
        num_prev_loans    = self.rng.integers(0, 8, size=n)

        # ── Target ────────────────────────────────────────────────────────────
        log_odds = (
            -3.0
            + 0.03  * (30 - age)
            - 0.005 * (credit_score - 650)
            + 2.5   * debt_to_income
            + 0.5   * loan_to_income
            + 0.15  * num_late_payments
            - 0.02  * (annual_income / 10_000)
            + 0.30  * (employment_type == "Unemployed").astype(float)
            - 0.20  * has_cosigner
            + self.rng.normal(0, 0.3, n)
        )
        prob_default = 1 / (1 + np.exp(-log_odds))
        default      = (self.rng.random(n) < prob_default).astype(int)

        df = pd.DataFrame({
            "age":                age,
            "education_level":    education,
            "marital_status":     marital_status,
            "num_dependents":     num_dependents,
            "region":             region,
            "employment_type":    employment_type,
            "employment_years":   employment_years.round(1),
            "months_employed":    months_employed,
            "annual_income":      annual_income.round(2),
            "monthly_income":     monthly_income.round(2),
            "savings_balance":    savings_balance.round(2),
            "investment_balance": investment_balance.round(2),
            "credit_score":       credit_score,
            "num_credit_lines":   num_credit_lines,
            "num_late_payments":  num_late_payments,
            "credit_util_rate":   credit_util_rate.round(4),
            "loan_purpose":       loan_purpose,
            "loan_amount":        loan_amount.round(2),
            "loan_term_months":   loan_term_months,
            "interest_rate":      interest_rate.round(2),
            "monthly_payment":    monthly_payment.round(2),
            "debt_to_income":     debt_to_income.round(4),
            "loan_to_income":     loan_to_income.round(4),
            "property_type":      property_type,
            "has_cosigner":       has_cosigner,
            "num_prev_loans":     num_prev_loans,
            "default":            default,
        })
        logger.info(f"Default rate: {default.mean():.2%}")
        return df

# ── Raw Data Pipeline ─────────────────────────────────────────────────────────
class RawDataPipeline(FeatureBuilder):

    def _save(self, df: pd.DataFrame, name: str) -> str:
        path = os.path.join(OUTPUT_DIR, f"{name}.csv")
        df.to_csv(path, index=False)
        logger.info(f"  Saved {name}.csv  ->  {df.shape[0]:,} rows x {df.shape[1]} cols")
        return path

    def run(self):
        logger.info("=== Raw Data Generation ===")

        raw = self.build(self.config.total_rows)
        raw = self._introduce_missing(raw, rate=0.03)
        raw = self._introduce_duplicates(raw, rate=0.005)
        raw = self._introduce_errors(raw)

        self._save(raw, "raw_data")
        logger.info(f"=== Done: {len(raw):,} rows saved to raw_data.csv ===")

# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg      = DataConfig(total_rows=100_000)
    pipeline = RawDataPipeline(cfg)
    pipeline.run()