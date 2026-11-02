"""Thread-safe in-memory request metrics for the API."""

from __future__ import annotations

import threading
import time
from typing import Any


class MetricsStore:
    """Collect request counters protected by a lock.

    The store is intentionally in-memory: metrics reset when the process
    restarts, which is acceptable for a demo service and documented in the
    README.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started_at = time.monotonic()
        self._total_requests = 0
        self._successful_predictions = 0
        self._failed_requests = 0
        self._total_response_time_ms = 0.0

    def record(self, duration_ms: float, status_code: int, is_prediction: bool) -> None:
        """Record one completed request.

        Args:
            duration_ms: Wall-clock handling time in milliseconds.
            status_code: Final HTTP status code.
            is_prediction: ``True`` for ``POST /predict`` requests.
        """
        with self._lock:
            self._total_requests += 1
            self._total_response_time_ms += max(duration_ms, 0.0)
            if is_prediction and 200 <= status_code < 300:
                self._successful_predictions += 1
            if status_code == 422 or status_code >= 500:
                self._failed_requests += 1

    def snapshot(self) -> dict[str, Any]:
        """Return a consistent snapshot of all metrics."""
        with self._lock:
            total = self._total_requests
            average = (self._total_response_time_ms / total) if total else 0.0
            return {
                "total_requests": total,
                "successful_predictions": self._successful_predictions,
                "failed_requests": self._failed_requests,
                "average_response_time_ms": round(average, 2),
                "uptime_seconds": round(time.monotonic() - self._started_at, 2),
            }

    def reset(self) -> None:
        """Reset counters (useful in tests)."""
        with self._lock:
            self._started_at = time.monotonic()
            self._total_requests = 0
            self._successful_predictions = 0
            self._failed_requests = 0
            self._total_response_time_ms = 0.0
