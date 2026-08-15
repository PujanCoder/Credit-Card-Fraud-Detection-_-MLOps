from dataclasses import dataclass
from pathlib import Path
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.exception import DataIngestionError
from src.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DataTransformationConfig:
    train_data_path: Path = PROJECT_ROOT / "artifacts" / "train.csv"
    test_data_path: Path = PROJECT_ROOT / "artifacts" / "test.csv"

    preprocessor_path: Path = PROJECT_ROOT / "artifacts" / "preprocessor.pkl"
    train_transformed_path: Path = PROJECT_ROOT / "artifacts" / "train_transformed.pkl"
    test_transformed_path: Path = PROJECT_ROOT / "artifacts" / "test_transformed.pkl"

    target_column: str = "is_fraud"


class DataTransformation:
    def __init__(self, config: DataTransformationConfig = DataTransformationConfig()):
        self.config = config

    @staticmethod
    def _haversine_distance_km(
        lat1: pd.Series,
        lon1: pd.Series,
        lat2: pd.Series,
        lon2: pd.Series,
    ) -> pd.Series:
        """Calculate straight-line distance between home and merchant."""
        radius_km = 6371.0

        lat1, lon1, lat2, lon2 = map(
            np.radians, [lat1, lon1, lat2, lon2]
        )

        delta_lat = lat2 - lat1
        delta_lon = lon2 - lon1

        a = (
            np.sin(delta_lat / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
        )
        return 2 * radius_km * np.arcsin(np.sqrt(a))

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        required_columns = {
            "trans_date_trans_time",
            "dob",
            "lat",
            "long",
            "merch_lat",
            "merch_long",
        }
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise DataIngestionError(
                f"Required columns are missing: {sorted(missing_columns)}"
            )

        transaction_time = pd.to_datetime(
            df["trans_date_trans_time"], errors="coerce"
        )
        date_of_birth = pd.to_datetime(df["dob"], errors="coerce")

        df["transaction_hour"] = transaction_time.dt.hour
        df["transaction_day_of_week"] = transaction_time.dt.dayofweek
        df["transaction_month"] = transaction_time.dt.month
        df["age"] = (transaction_time - date_of_birth).dt.days / 365.25
        df["age"] = df["age"].clip(lower=18, upper=100)

        df["merchant_distance_km"] = self._haversine_distance_km(
            df["lat"],
            df["long"],
            df["merch_lat"],
            df["merch_long"],
        )

        # Drop identifiers, direct PII, source date fields, and raw coordinates.
        columns_to_drop = [
            "Unnamed: 0",
            "cc_num",
            "first",
            "last",
            "gender",
            "street",
            "zip",
            "dob",
            "trans_num",
            "trans_date_trans_time",
            "unix_time",
            "lat",
            "long",
            "merch_lat",
            "merch_long",
        ]
        return df.drop(columns=columns_to_drop, errors="ignore")

    @staticmethod
    def _build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
        categorical_columns = X_train.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        numerical_columns = X_train.select_dtypes(
            include=["number", "bool"]
        ).columns.tolist()

        numeric_pipeline = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, numerical_columns),
                ("categorical", categorical_pipeline, categorical_columns),
            ],
            remainder="drop",
        )

    def initiate_data_transformation(self) -> tuple[Path, Path, Path]:
        try:
            logger.info("Data transformation started.")

            train_df = pd.read_csv(self.config.train_data_path)
            test_df = pd.read_csv(self.config.test_data_path)

            logger.info(
                "Input loaded safely: train_rows=%d, test_rows=%d",
                len(train_df),
                len(test_df),
            )

            target = self.config.target_column
            if target not in train_df.columns or target not in test_df.columns:
                raise DataIngestionError(f"Target column '{target}' is missing.")

            y_train = train_df[target].astype(int)
            y_test = test_df[target].astype(int)

            X_train = self.create_features(train_df.drop(columns=[target]))
            X_test = self.create_features(test_df.drop(columns=[target]))

            preprocessor = self._build_preprocessor(X_train)

            logger.info("Fitting preprocessor using training data only.")
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            self.config.preprocessor_path.parent.mkdir(
                parents=True, exist_ok=True
            )

            joblib.dump(preprocessor, self.config.preprocessor_path)

            joblib.dump(
                {"X": X_train_transformed, "y": y_train.to_numpy()},
                self.config.train_transformed_path,
            )
            joblib.dump(
                {"X": X_test_transformed, "y": y_test.to_numpy()},
                self.config.test_transformed_path,
            )

            logger.info(
                "Transformation completed: train_features=%d, test_features=%d",
                X_train_transformed.shape[1],
                X_test_transformed.shape[1],
            )
            logger.info("Preprocessor and transformed artifacts saved.")

            return (
                self.config.train_transformed_path,
                self.config.test_transformed_path,
                self.config.preprocessor_path,
            )

        except FileNotFoundError as exc:
            logger.exception("Required ingestion artifact was not found.")
            raise DataIngestionError("Run data ingestion before transformation.") from exc

        except Exception as exc:
            logger.exception("Data transformation failed.")
            raise DataIngestionError("Data transformation failed.") from exc


if __name__ == "__main__":
    transformer = DataTransformation()

    train_path, test_path, preprocessor_path = (
        transformer.initiate_data_transformation()
    )

    print(f"Train artifact: {train_path}")
    print(f"Test artifact: {test_path}")
    print(f"Preprocessor: {preprocessor_path}")
