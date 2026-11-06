from __future__ import annotations

import json
import logging

from app.logging_config import JsonFormatter


def test_json_incluye_request_id_y_metricas():
    record = logging.LogRecord("app.main", logging.INFO, __file__, 1, "analyze", None, None)
    record.request_id = "req-1"
    record.inference_ms = 120
    data = json.loads(JsonFormatter().format(record))
    assert data["message"] == "analyze"
    assert data["request_id"] == "req-1"
    assert data["inference_ms"] == 120
    assert data["level"] == "INFO"
