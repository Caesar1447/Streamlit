"""Privacy-conscious JSON request logs for the educational FDE API.

Do not log patient data, request bodies, headers, query strings or IP addresses.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request

_LOG = logging.getLogger("fde.api.requests")
if not _LOG.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _LOG.addHandler(_handler)
_LOG.setLevel(logging.INFO)
_LOG.propagate = False

# Do not record arbitrary URL paths: they could contain identifying values.
_SAFE_ENDPOINTS = frozenset(
    {
        "/health",
        "/docs",
        "/openapi.json",
        "/model/info",
        "/predict",
        "/benchmarks/uci/summary",
        "/benchmarks/uci/evaluation",
    }
)


def install_request_monitoring(app: FastAPI) -> None:
    """Register request timing and redacted, structured access logs."""

    @app.middleware("http")
    async def monitor_request(request: Request, call_next):
        request_id = uuid4().hex
        started = perf_counter()
        status_code = 500
        error_type = None

        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time-Ms"] = f"{(perf_counter() - started) * 1000:.2f}"
            return response
        except Exception as exc:
            # Exception type only; exception messages can contain sensitive data.
            error_type = type(exc).__name__
            raise
        finally:
            path = request.url.path
            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": "http_request",
                "request_id": request_id,
                "method": request.method,
                "endpoint": path if path in _SAFE_ENDPOINTS else "[redacted]",
                "status_code": status_code,
                "duration_ms": round((perf_counter() - started) * 1000, 2),
            }
            if error_type is not None:
                event["error_type"] = error_type
            _LOG.info(json.dumps(event, separators=(",", ":")))
