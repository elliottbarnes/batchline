"""Bounded queue and dynamic batching worker."""

from concurrent.futures import Future
from dataclasses import dataclass
import queue
import threading
import time
from typing import Dict, List, Sequence

from .metrics import Metrics


class QueueFull(Exception):
    """Raised when backpressure rejects a request."""


@dataclass
class WorkItem:
    inputs: Sequence[float]
    future: Future


class BatchExecutor:
    def __init__(
        self,
        model: object,
        metrics: Metrics,
        max_batch_size: int,
        batch_window_ms: int,
        max_queue_size: int,
    ) -> None:
        self.model = model
        self.metrics = metrics
        self.max_batch_size = max_batch_size
        self.batch_window_seconds = batch_window_ms / 1000
        self._queue: "queue.Queue[WorkItem]" = queue.Queue(maxsize=max_queue_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._work, name="batch-worker", daemon=True)
        self._thread.start()

    @property
    def ready(self) -> bool:
        return self._thread.is_alive() and not self._stop.is_set()

    def submit(self, inputs: Sequence[float]) -> Future:
        future: Future = Future()
        try:
            self._queue.put_nowait(WorkItem(inputs=inputs, future=future))
        except queue.Full as exc:
            raise QueueFull from exc
        self.metrics.queue_depth(self._queue.qsize())
        return future

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _work(self) -> None:
        while not self._stop.is_set():
            try:
                first = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            batch = [first]
            deadline = time.monotonic() + self.batch_window_seconds
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._queue.get(timeout=remaining))
                except queue.Empty:
                    break

            self.metrics.queue_depth(self._queue.qsize())
            started = time.monotonic()
            try:
                outputs: List[Dict[str, object]] = self.model.predict_batch(
                    [item.inputs for item in batch]
                )
                if len(outputs) != len(batch):
                    raise RuntimeError("model returned a different number of outputs than inputs")
                for item, output in zip(batch, outputs):
                    item.future.set_result(output)
            except Exception as exc:  # keep one model failure from killing the worker
                for item in batch:
                    item.future.set_exception(exc)
            finally:
                elapsed = time.monotonic() - started
                self.metrics.observe_inference(elapsed, len(batch))
                for _ in batch:
                    self._queue.task_done()
