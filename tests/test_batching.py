import threading
import unittest

from inference_service.batching import BatchExecutor, QueueFull
from inference_service.metrics import Metrics


class RecordingModel:
    def __init__(self) -> None:
        self.batch_sizes = []
        self.lock = threading.Lock()

    def predict_batch(self, batch):
        with self.lock:
            self.batch_sizes.append(len(batch))
        return [{"value": values[0]} for values in batch]


class BlockingModel:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def predict_batch(self, batch):
        self.started.set()
        self.release.wait(timeout=1)
        return [{"value": values[0]} for values in batch]


class BatchExecutorTests(unittest.TestCase):
    def test_combines_concurrent_work_into_a_batch(self) -> None:
        model = RecordingModel()
        executor = BatchExecutor(model, Metrics(), max_batch_size=4, batch_window_ms=100, max_queue_size=8)
        try:
            futures = [executor.submit([number]) for number in range(4)]
            outputs = [future.result(timeout=1) for future in futures]
            self.assertEqual([output["value"] for output in outputs], [0, 1, 2, 3])
            self.assertEqual(model.batch_sizes, [4])
        finally:
            executor.close()

    def test_rejects_work_when_bounded_queue_is_full(self) -> None:
        model = BlockingModel()
        executor = BatchExecutor(model, Metrics(), max_batch_size=1, batch_window_ms=0, max_queue_size=1)
        try:
            first = executor.submit([1])
            self.assertTrue(model.started.wait(timeout=1))
            second = executor.submit([2])
            with self.assertRaises(QueueFull):
                executor.submit([3])
            model.release.set()
            self.assertEqual(first.result(timeout=1)["value"], 1)
            self.assertEqual(second.result(timeout=1)["value"], 2)
        finally:
            model.release.set()
            executor.close()


if __name__ == "__main__":
    unittest.main()
