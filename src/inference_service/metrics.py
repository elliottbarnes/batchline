"""Small, thread-safe Prometheus text-format metrics registry."""

from collections import defaultdict
import threading
from typing import DefaultDict, Dict, Iterable, Tuple


LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: DefaultDict[Tuple[str, str], int] = defaultdict(int)
        self._inference_count = 0
        self._inference_sum = 0.0
        self._inference_buckets: Dict[float, int] = {bucket: 0 for bucket in LATENCY_BUCKETS}
        self._batches = 0
        self._batch_items = 0
        self._queue_depth = 0

    def request(self, method: str, status: int) -> None:
        with self._lock:
            self._requests[(method, str(status))] += 1

    def observe_inference(self, seconds: float, batch_size: int) -> None:
        with self._lock:
            self._inference_count += 1
            self._inference_sum += seconds
            self._batches += 1
            self._batch_items += batch_size
            for bucket in LATENCY_BUCKETS:
                if seconds <= bucket:
                    self._inference_buckets[bucket] += 1

    def queue_depth(self, value: int) -> None:
        with self._lock:
            self._queue_depth = value

    def render(self) -> str:
        with self._lock:
            requests = dict(self._requests)
            inference_count = self._inference_count
            inference_sum = self._inference_sum
            inference_buckets = dict(self._inference_buckets)
            batches = self._batches
            batch_items = self._batch_items
            queue_depth = self._queue_depth

        lines = [
            "# HELP inference_http_requests_total HTTP requests handled.",
            "# TYPE inference_http_requests_total counter",
        ]
        for (method, status), value in sorted(requests.items()):
            lines.append(
                f'inference_http_requests_total{{method="{method}",status="{status}"}} {value}'
            )
        lines.extend(
            [
                "# HELP inference_queue_depth Requests waiting for a worker.",
                "# TYPE inference_queue_depth gauge",
                f"inference_queue_depth {queue_depth}",
                "# HELP inference_batches_total Model batches executed.",
                "# TYPE inference_batches_total counter",
                f"inference_batches_total {batches}",
                "# HELP inference_batch_items_total Items processed in model batches.",
                "# TYPE inference_batch_items_total counter",
                f"inference_batch_items_total {batch_items}",
                "# HELP inference_duration_seconds Time spent executing model batches.",
                "# TYPE inference_duration_seconds histogram",
            ]
        )
        for bucket in LATENCY_BUCKETS:
            lines.append(
                f'inference_duration_seconds_bucket{{le="{bucket:g}"}} '
                f"{inference_buckets[bucket]}"
            )
        lines.extend(
            [
                f'inference_duration_seconds_bucket{{le="+Inf"}} {inference_count}',
                f"inference_duration_seconds_sum {inference_sum:.9f}",
                f"inference_duration_seconds_count {inference_count}",
            ]
        )
        return "\n".join(lines) + "\n"
