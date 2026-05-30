import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class InteractionFeatures:
    """Creates interaction and ratio features from top correlated columns."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"\n{'='*55}")
        print(f"  INTERACTION FEATURES")
        print(f"{'='*55}")

        before = df.shape[1]

        # 1. Combined debt burden
        df["debt_x_loan"] = df["debt_to_income"] * df["loan_to_income"]
        print(f"  [+] debt_x_loan          = debt_to_income x loan_to_income")

        # 2. Repayment capacity
        df["income_per_loan"] = df["annual_income"] / (df["loan_amount"] + 1e-9)
        print(f"  [+] income_per_loan      = annual_income / loan_amount")

        # 3. Credit risk interaction
        df["score_x_dti"] = df["credit_score"] * df["debt_to_income"]
        print(f"  [+] score_x_dti          = credit_score x debt_to_income")

        # 4. Savings buffer ratio
        df["savings_to_loan"] = df["savings_balance"] / (df["loan_amount"] + 1e-9)
        print(f"  [+] savings_to_loan      = savings_balance / loan_amount")

        # 5. Behavioral risk
        df["late_x_dti"] = df["num_late_payments"] * df["debt_to_income"]
        print(f"  [+] late_x_dti           = num_late_payments x debt_to_income")

        # 6. Stability signal
        df["age_x_income"] = df["age"] * df["annual_income"]
        print(f"  [+] age_x_income         = age x annual_income")

        # 7. Total burden
        df["loan_burden"] = df["loan_amount"] / (df["annual_income"] + df["savings_balance"] + 1e-9)
        print(f"  [+] loan_burden          = loan_amount / (annual_income + savings_balance)")

        after = df.shape[1]
        print(f"\n  Features added : {after - before}")
        print(f"  Total features : {after}")
        print(f"{'='*55}\n")

        logger.info(f"[InteractionFeatures] Added {after - before} features | Total: {after}")
        return df