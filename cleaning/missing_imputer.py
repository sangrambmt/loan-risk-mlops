import pandas as pd
import logging

logger = logging.getLogger(__name__)

class MissingImputer:
    """Imputes missing values using median for numerics and mode for categoricals."""

    NUMERIC_COLS = [
        "age", "employment_years", "months_employed", "annual_income",
        "monthly_income", "savings_balance", "investment_balance",
        "credit_score", "num_credit_lines", "num_late_payments",
        "credit_util_rate", "loan_amount", "loan_term_months",
        "interest_rate", "monthly_payment", "debt_to_income",
        "loan_to_income", "num_dependents", "has_cosigner", "num_prev_loans"
    ]

    CATEGORICAL_COLS = [
        "education_level", "marital_status", "region",
        "employment_type", "loan_purpose", "property_type"
    ]

    def __init__(self):
        self._num_fill: dict = {}
        self._cat_fill: dict = {}

    def fit(self, df: pd.DataFrame) -> "MissingImputer":
        for col in self.NUMERIC_COLS:
            if col in df.columns:
                self._num_fill[col] = df[col].median()
        for col in self.CATEGORICAL_COLS:
            if col in df.columns:
                self._cat_fill[col] = df[col].mode()[0]
        logger.info(f"[MissingImputer] Fitted on {len(df):,} rows")
        print(f"[MissingImputer] Fitted on {len(df):,} rows")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        before = df.isnull().sum().sum()
        df     = df.fillna({**self._num_fill, **self._cat_fill})
        after  = df.isnull().sum().sum()
        logger.info(f"[MissingImputer] Filled {before - after:,} missing values")
        print(f"[MissingImputer] Filled {before - after:,} missing values | Remaining nulls: {after:,}")
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)