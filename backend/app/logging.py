import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "api",
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        if hasattr(record, "run_id"):
            payload["run_id"] = record.run_id
        if hasattr(record, "stage"):
            payload["stage"] = record.stage
        if hasattr(record, "extra"):
            payload["extra"] = record.extra
        return json.dumps(payload, ensure_ascii=True)


def configure_logging(service: str = "api") -> None:
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    formatter = JsonFormatter()
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    original_factory = logging.getLogRecordFactory()

    def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = original_factory(*args, **kwargs)
        setattr(record, "service", service)
        return record

    logging.setLogRecordFactory(record_factory)
