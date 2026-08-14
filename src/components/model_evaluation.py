from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import json
import logging

import joblib
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.exception import ModelEvaluationError
from src.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelEvaluationConfig:
    test_transformed_path: Path = (
        PROJECT_ROOT / "artifacts" / "test_transformed.pkl"
    )
    model_path: Path = PROJECT_ROOT / "models" / "fraud_model.pkl"

    metrics_path: Path = PROJECT_ROOT / "reports" / "test_metrics.json"
    classification_report_path: Path = (
        PROJECT_ROOT / "reports" / "classification_report.txt"
    )

    fraud_threshold: float = 0.50


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig = ModelEvaluationConfig()):
        self.config = config

    @staticmethod
    def _progress(message: str) -> None:
        print(message, flush=True)
        logger.info(message)

    def initiate_model_evaluation(self) -> Path:
        start_time = perf_counter()

        try:
            self._progress("[1/4] Loading trained model...")
            model = joblib.load(self.config.model_path)

            self._progress("[2/4] Loading untouched test data...")
            test_artifact = joblib.load(self.config.test_transformed_path)
            X_test = test_artifact["X"]
            y_test = test_artifact["y"]

            self._progress(
                f"[3/4] Generating predictions for {X_test.shape[0]:,} test records..."
            )
            probabilities = model.predict_proba(X_test)[:, 1]
            predictions = (
                probabilities >= self.config.fraud_threshold
            ).astype(int)

            tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

            metrics = {
                "roc_auc": float(roc_auc_score(y_test, probabilities)),
                "pr_auc": float(average_precision_score(y_test, probabilities)),
                "precision": float(
                    precision_score(y_test, predictions, zero_division=0)
                ),
                "recall": float(
                    recall_score(y_test, predictions, zero_division=0)
                ),
                "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
                "fraud_threshold": self.config.fraud_threshold,
                "test_rows": int(X_test.shape[0]),
                "actual_fraud_cases": int(y_test.sum()),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            }

            report = classification_report(
                y_test,
                predictions,
                target_names=["legitimate", "fraud"],
                zero_division=0,
            )

            self.config.metrics_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.config.metrics_path, "w", encoding="utf-8") as file:
                json.dump(metrics, file, indent=4)

            with open(
                self.config.classification_report_path, "w", encoding="utf-8"
            ) as file:
                file.write(report)

            elapsed_seconds = perf_counter() - start_time

            self._progress(
                "[4/4] Evaluation complete | "
                f"PR-AUC={metrics['pr_auc']:.4f} | "
                f"Precision={metrics['precision']:.4f} | "
                f"Recall={metrics['recall']:.4f} | "
                f"Time={elapsed_seconds:.1f} seconds"
            )

            self._progress(
                f"Confusion matrix: TP={tp:,}, FP={fp:,}, FN={fn:,}, TN={tn:,}"
            )

            return self.config.metrics_path

        except FileNotFoundError as exc:
            logger.exception("A required model or test artifact was not found.")
            raise ModelEvaluationError(
                "Run data transformation and model training before evaluation."
            ) from exc

        except Exception as exc:
            logger.exception("Model evaluation failed.")
            raise ModelEvaluationError("Model evaluation failed.") from exc


if __name__ == "__main__":
    evaluator = ModelEvaluation()
    metrics_path = evaluator.initiate_model_evaluation()
    print(f"\nFinal test metrics saved to: {metrics_path}")