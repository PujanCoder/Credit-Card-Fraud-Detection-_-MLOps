from functools import lru_cache
from pathlib import Path
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from src.components.data_transformation import DataTransformation


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.pkl"
PREPROCESSOR_PATH = PROJECT_ROOT / "artifacts" / "preprocessor.pkl"

FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.50"))

app = Flask(__name__)

REQUIRED_FIELDS = {
    "trans_date_trans_time",
    "merchant",
    "category",
    "amt",
    "city",
    "state",
    "city_pop",
    "job",
    "dob",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
}

NUMERIC_FIELDS = {
    "amt",
    "city_pop",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
}


@lru_cache(maxsize=1)
def load_artifacts():
    """Load the trained model and fitted preprocessor once."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model missing: {MODEL_PATH}")

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor missing: {PREPROCESSOR_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    return model, preprocessor


def validate_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    missing_fields = REQUIRED_FIELDS - payload.keys()
    if missing_fields:
        raise ValueError(
            f"Missing required fields: {sorted(missing_fields)}"
        )

    record = dict(payload)

    for field in NUMERIC_FIELDS:
        try:
            record[field] = float(record[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"'{field}' must be a valid number."
            ) from exc

    for field in ("trans_date_trans_time", "dob"):
        if pd.isna(pd.to_datetime(record[field], errors="coerce")):
            raise ValueError(
                f"'{field}' must be a valid date or datetime."
            )

    return record

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Credit Card Fraud Detection API is running.",
        "health_endpoint": "/health",
        "prediction_endpoint": "/predict",
    }), 200

@app.get("/health")
def health():
    try:
        load_artifacts()
        return jsonify({
            "status": "healthy",
            "model_loaded": True,
        }), 200
    except Exception:
        app.logger.exception("Health check failed.")
        return jsonify({
            "status": "unhealthy",
            "model_loaded": False,
        }), 503


@app.post("/predict")
def predict():
    try:
        payload = request.get_json(silent=True)
        record = validate_payload(payload)

        model, preprocessor = load_artifacts()

        raw_df = pd.DataFrame([record])

        # Reuses the same feature engineering used during training.
        feature_builder = DataTransformation()
        features = feature_builder.create_features(raw_df)

        transformed_features = preprocessor.transform(features)
        fraud_probability = float(
            model.predict_proba(transformed_features)[0, 1]
        )

        prediction = (
            "fraud"
            if fraud_probability >= FRAUD_THRESHOLD
            else "legitimate"
        )

        # Do not log request JSON or cardholder information.
        app.logger.info("Prediction served successfully.")

        return jsonify({
            "prediction": prediction,
            "fraud_probability": round(fraud_probability, 6),
            "fraud_threshold": FRAUD_THRESHOLD,
        }), 200

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    except FileNotFoundError as exc:
        app.logger.exception("Model artifact is unavailable.")
        return jsonify({"error": str(exc)}), 503

    except Exception:
        app.logger.exception("Prediction failed.")
        return jsonify({"error": "Internal prediction error."}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)