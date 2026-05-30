import pandas as pd
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cleaning.duplicate_remover import DuplicateRemover
from cleaning.error_corrector   import ErrorCorrector
from cleaning.column_dropper    import ColumnDropper
from cleaning.missing_imputer   import MissingImputer
from cleaning.type_enforcer     import TypeEnforcer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

class DataCleaningPipeline:
    """Orchestrates all cleaning steps sequentially."""

    def __init__(self):
        self.steps = [
            DuplicateRemover(),
            ErrorCorrector(),
            ColumnDropper(threshold=0.50),
            MissingImputer(),
            TypeEnforcer(),
        ]

    def _summary(self, df: pd.DataFrame, label: str):
        print(f"\n{'='*50}")
        print(f"  {label}")
        print(f"{'='*50}")
        print(f"  Rows        : {len(df):,}")
        print(f"  Columns     : {df.shape[1]}")
        print(f"  Nulls       : {df.isnull().sum().sum():,}")
        print(f"  Duplicates  : {df.duplicated().sum():,}")
        print(f"{'='*50}\n")

    def run(self):
        input_path = os.path.join(DATA_DIR, "raw_data.csv")
        logger.info(f"Loading raw data from {input_path}")
        df = pd.read_csv(input_path)
        self._summary(df, "BEFORE CLEANING")

        for step in self.steps:
            df = step.fit_transform(df) if hasattr(step, "fit_transform") else step.transform(df)

        self._summary(df, "AFTER CLEANING")

        output_path = os.path.join(DATA_DIR, "clean_data.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Clean data saved -> {output_path}")
        print(f"\n  Clean data saved -> {output_path}\n")

if __name__ == "__main__":
    DataCleaningPipeline().run()