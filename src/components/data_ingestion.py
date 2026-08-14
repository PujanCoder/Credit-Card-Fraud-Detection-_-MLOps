from dataclasses import dataclass
from pathlib import Path
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import DataIngestionError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataIngestionConfig:
    raw_data_path: Path = Path("data/fraudTrain.csv")
    artifact_dir: Path = Path("artifacts")
    train_data_path: Path = Path("artifacts/train.csv")
    test_data_path: Path = Path("artifacts/test.csv")
    test_size: float = 0.20
    random_state: int = 42
    target_column: str = "is_fraud"


class DataIngestion:
    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()):
        self.config = config

    def _validate_data(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise DataIngestionError("Dataset is empty.")

        if self.config.target_column not in df.columns:
            raise DataIngestionError(
                f"Target column '{self.config.target_column}' is missing."
            )

        invalid_labels = set(df[self.config.target_column].dropna().unique()) - {0, 1}
        if invalid_labels:
            raise DataIngestionError(
                f"Unexpected target values found: {invalid_labels}"
            )

    def initiate_data_ingestion(self) -> tuple[Path, Path]:
        """Load, validate, split, and persist the fraud dataset."""
        try:
            logger.info("Data ingestion started.")
            logger.info("Reading raw dataset from %s", self.config.raw_data_path)

            df = pd.read_csv(self.config.raw_data_path)
            logger.info("Raw data loaded: rows=%d, columns=%d", *df.shape)

            self._validate_data(df)

            logger.info(
                "Data validation passed. fraud_rate=%.4f, missing_values=%d",
                df[self.config.target_column].mean(),
                int(df.isna().sum().sum()),
            )

            self.config.artifact_dir.mkdir(parents=True, exist_ok=True)

            train_df, test_df = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=df[self.config.target_column],
            )

            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)

            logger.info(
                "Data split saved successfully: train_rows=%d, test_rows=%d",
                len(train_df),
                len(test_df),
            )
            logger.info(
                "Artifacts saved: train=%s, test=%s",
                self.config.train_data_path,
                self.config.test_data_path,
            )

            return self.config.train_data_path, self.config.test_data_path

        except FileNotFoundError as exc:
            logger.exception("Raw-data file was not found.")
            raise DataIngestionError(
                f"Could not find dataset at: {self.config.raw_data_path}"
            ) from exc

        except Exception as exc:
            logger.exception("Data ingestion failed.")
            raise DataIngestionError("Data ingestion failed.") from exc


if __name__ == "__main__":
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    print(f"Train data: {train_path}")
    print(f"Test data: {test_path}")