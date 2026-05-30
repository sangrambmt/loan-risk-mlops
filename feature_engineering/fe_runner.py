import pandas as pd
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from feature_engineering.feature_selector    import FeatureSelector
from feature_engineering.encoder             import Encoder
from feature_engineering.scaler              import Scaler
from feature_engineering.splitter            import Splitter
from feature_engineering.interaction_features import InteractionFeatures
from feature_engineering.balancer            import Balancer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

class FeatureEngineeringRunner:
    """Orchestrates all feature engineering steps sequentially."""

    def __init__(self):
        self.selector     = FeatureSelector()
        self.interactions = InteractionFeatures()
        self.encoder      = Encoder()
        self.scaler       = Scaler()
        self.splitter     = Splitter(test_size=0.15, val_size=0.15,
                                     target_col="default", seed=42)
        self.balancer     = Balancer(target_col="default",
                                     sampling_strategy=0.40, seed=42)

    def _summary(self, df: pd.DataFrame, label: str):
        print(f"\n{'='*55}")
        print(f"  {label}")
        print(f"{'='*55}")
        print(f"  Rows    : {len(df):,}")
        print(f"  Columns : {df.shape[1]}")
        print(f"  Nulls   : {df.isnull().sum().sum():,}")
        print(f"{'='*55}\n")

    def run(self):
        logger.info("=== Feature Engineering Pipeline Started ===")

        # 1. Load
        path = os.path.join(DATA_DIR, "clean_data.csv")
        logger.info(f"Loading clean data from {path}")
        df   = pd.read_csv(path)
        self._summary(df, "INPUT DATA")

        # 2. Feature Selection
        logger.info("Step 1: Feature Selection ...")
        df = self.selector.transform(df)
        self._summary(df, "AFTER FEATURE SELECTION")

        # 3. Interaction Features
        logger.info("Step 2: Interaction Features ...")
        df = self.interactions.transform(df)
        self._summary(df, "AFTER INTERACTION FEATURES")

        # 4. Encode
        logger.info("Step 3: Encoding ...")
        df = self.encoder.fit_transform(df)
        self.encoder.save()
        self._summary(df, "AFTER ENCODING")

        # 5. Scale
        logger.info("Step 4: Scaling ...")
        df = self.scaler.fit_transform(df)
        self.scaler.save()
        self._summary(df, "AFTER SCALING")

        # 6. Split
        logger.info("Step 5: Splitting ...")
        train, val, test = self.splitter.split(df)

        # 7. SMOTE on train only
        logger.info("Step 6: SMOTE Balancing on train set ...")
        train_balanced = self.balancer.fit_transform(train)
        out = os.path.join(DATA_DIR, "train_balanced.csv")
        train_balanced.to_csv(out, index=False)
        logger.info(f"Balanced train saved -> {out}")
        print(f"  Balanced train saved -> {out}")

        logger.info("=== Feature Engineering Pipeline Complete ===")
        print(f"\n  Artifacts -> feature_engineering/artifacts/")
        print(f"  Data      -> data/train_balanced.csv, val_data.csv, test_data.csv\n")

if __name__ == "__main__":
    FeatureEngineeringRunner().run()