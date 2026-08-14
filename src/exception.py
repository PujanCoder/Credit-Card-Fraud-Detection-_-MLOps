class DataIngestionError(Exception):
    """Raised when the data-ingestion component cannot complete."""
class ModelTrainingError(Exception):
    """Raised when model training cannot complete."""
class ModelEvaluationError(Exception):
    """Raised when final model evaluation cannot complete."""