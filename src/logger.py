import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path


class PiiRedactionFilter(logging.Filter):
    """Redacts common PII patterns from log messages."""

    CARD_NUMBER_PATTERN = re.compile(r"\b(?:\d[ -]?){13,19}\b")
    SENSITIVE_FIELD_PATTERN = re.compile(
        r"(?i)\b(cc_num|first|last|street|zip|dob|trans_num)\b\s*[:=]\s*[^,;\s]+"
    )

    def filter(self, record: logging.LogRecord) -> bool:
        # Resolve %s arguments first, then store only the redacted message.
        message = record.getMessage()
        message = self.CARD_NUMBER_PATTERN.sub("[REDACTED_CARD]", message)
        message = self.SENSITIVE_FIELD_PATTERN.sub(r"\1=[REDACTED]", message)

        record.msg = message
        record.args = ()
        return True


def configure_logging() -> None:
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if root_logger.handlers:
        return

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    redaction_filter = PiiRedactionFilter()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(redaction_filter)

    file_handler = RotatingFileHandler(
        log_dir / "pipeline.log",
        maxBytes=5_000_000,
        backupCount=5,
        encoding="utf-8",
        delay=True,
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(redaction_filter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)