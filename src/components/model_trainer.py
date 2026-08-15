from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import json
import logging
import os
import mlflow
import mlflow.sklearn


mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
)
mlflow.set_experiment(
    os.getenv("MLFLOW_EXPERIMENT_NAME", "credit-card-fraud-detection")
)


import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.exception import ModelTrainingError
from src.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelTrainerConfig:
    train_transformed_path: Path = (
        PROJECT_ROOT / "artifacts" / "train_transformed.pkl"
    )
    model_path: Path = PROJECT_ROOT / "models" / "fraud_model.pkl"
    metrics_path: Path = PROJECT_ROOT / "reports" / "validation_metrics.json"

    validation_size: float = 0.20
    random_state: int = 42
    total_trees: int = 100
    trees_per_batch: int = 20
    fraud_threshold: float = 0.50


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig = ModelTrainerConfig()):
        self.config = config

    @staticmethod
    def _progress(message: str) -> None:
        """Print progress for the terminal and record it in the log."""
        print(message, flush=True)
        logger.info(message)

    def initiate_model_training(self) -> Path:
        start_time = perf_counter()


        try:
            self._progress("[1/5] Loading transformed training data...")

            train_artifact = joblib.load(self.config.train_transformed_path)
            X = train_artifact["X"]
            y = train_artifact["y"]

            self._progress(
                f"[2/5] Data loaded: {X.shape[0]:,} rows, "
                f"{X.shape[1]:,} features."
            )

            X_train, X_validation, y_train, y_validation = train_test_split(
                X,
                y,
                test_size=self.config.validation_size,
                random_state=self.config.random_state,
                stratify=y,
            )

            self._progress(
                f"[3/5] Split complete: train={X_train.shape[0]:,}, "
                f"validation={X_validation.shape[0]:,}."
            )

            model = RandomForestClassifier(
                n_estimators=self.config.trees_per_batch,
                max_depth=18,
                min_samples_leaf=10,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=self.config.random_state,
                n_jobs=-1,
                warm_start=True,
            )

            self._progress(
                f"[4/5] Training Random Forest with "
                f"{self.config.total_trees} trees..."
            )

            for tree_count in range(
                self.config.trees_per_batch,
                self.config.total_trees + 1,
                self.config.trees_per_batch,
            ):
                model.set_params(n_estimators=tree_count)
                model.fit(X_train, y_train)

                completion = (tree_count / self.config.total_trees) * 100
                elapsed_seconds = perf_counter() - start_time

                self._progress(
                    f"      Training progress: {completion:.0f}% "
                    f"({tree_count}/{self.config.total_trees} trees) | "
                    f"elapsed: {elapsed_seconds / 60:.1f} minutes"
                )

            self._progress("[5/5] Calculating validation metrics...")

            probabilities = model.predict_proba(X_validation)[:, 1]
            predictions = (
                probabilities >= self.config.fraud_threshold
            ).astype(int)

            metrics = {
                "roc_auc": float(roc_auc_score(y_validation, probabilities)),
                "pr_auc": float(
                    average_precision_score(y_validation, probabilities)
                ),
                "precision": float(precision_score(
                    y_validation, predictions, zero_division=0
                )),
                "recall": float(recall_score(
                    y_validation, predictions, zero_division=0
                )),
                "f1_score": float(f1_score(
                    y_validation, predictions, zero_division=0
                )),
                "fraud_threshold": self.config.fraud_threshold,
                "training_rows": int(X_train.shape[0]),
                "validation_rows": int(X_validation.shape[0]),
                "feature_count": int(X_train.shape[1]),
            }

            self.config.model_path.parent.mkdir(parents=True, exist_ok=True)
            self.config.metrics_path.parent.mkdir(parents=True, exist_ok=True)

            joblib.dump(model, self.config.model_path, compress=3)

            with open(self.config.metrics_path, "w", encoding="utf-8") as file:
                json.dump(metrics, file, indent=4)

            total_minutes = (perf_counter() - start_time) / 60

            self._progress(
                "Training complete | "
                f"PR-AUC={metrics['pr_auc']:.4f} | "
                f"Recall={metrics['recall']:.4f} | "
                f"Total time={total_minutes:.1f} minutes"
            )

            return self.config.model_path
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "credit-card-fraud-detection"))

        except FileNotFoundError as exc:
            logger.exception("Transformed training artifact was not found.")
            raise ModelTrainingError(
                "Run data transformation before model training."
            ) from exc

        except Exception as exc:
            logger.exception("Model training failed.")
            raise ModelTrainingError("Model training failed.") from exc


if __name__ == "__main__":
    trainer = ModelTrainer()
    model_path = trainer.initiate_model_training()
    print(f"\nModel saved successfully: {model_path}")