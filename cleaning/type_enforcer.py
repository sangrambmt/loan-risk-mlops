import pandas as pd
import logging

logger = logging.getLogger(__name__)

class TypeEnforcer:
    """Enforces correct data types for all columns."""

    INT_COLS = [
        "age", "credit_score", "num_credit_lines", "num_late_payments",
        "loan_term_months", "months_employed", "num_dependents",
        "has_cosigner", "num_prev_loans"
    ]

    FLOAT_COLS = [
        "annual_income", "monthly_income", "savings_balance",
        "investment_balance", "loan_amount", "interest_rate",
        "monthly_payment", "debt_to_income", "loan_to_income",
        "credit_util_rate", "employment_years"
    ]

    STR_COLS = [
        "education_level", "marital_status", "region",
        "employment_type", "loan_purpose", "property_type"
    ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.INT_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
                logger.info(f"[TypeEnforcer] {col} -> int")
                print(f"[TypeEnforcer] {col} -> int")

        for col in self.FLOAT_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
                logger.info(f"[TypeEnforcer] {col} -> float")
                print(f"[TypeEnforcer] {col} -> float")

        for col in self.STR_COLS:
            if col in df.columns:
                df[col] = df[col].astype(str)
                logger.info(f"[TypeEnforcer] {col} -> string")
                print(f"[TypeEnforcer] {col} -> string")

        return df