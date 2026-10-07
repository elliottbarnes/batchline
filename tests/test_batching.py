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

class LifecycleTests(unittest.TestCase):
    def test_close_rejects_new_work_and_resolves_waiting_work(self):
        from inference_service.batching import ExecutorClosed
        model = BlockingModel()
        executor = BatchExecutor(model, Metrics(), 1, 0, 8)
        first = executor.submit([1])
        self.assertTrue(model.started.wait(1))
        waiting = executor.submit([2])
        closer = threading.Thread(target=executor.close)
        closer.start()
        try:
            with self.assertRaises(ExecutorClosed):
                waiting.result(timeout=1)
            with self.assertRaises(ExecutorClosed):
                executor.submit([3])
            self.assertFalse(executor.ready)
        finally:
            model.release.set()
            closer.join(timeout=3)
        self.assertEqual(first.result(timeout=1), {"value": 1})
        executor.close()  # Idempotent.

    def test_cancelled_queued_future_does_not_kill_worker(self):
        model = BlockingModel()
        executor = BatchExecutor(model, Metrics(), 1, 0, 8)
        try:
            first = executor.submit([1])
            self.assertTrue(model.started.wait(1))
            cancelled = executor.submit([2])
            self.assertTrue(cancelled.cancel())
            model.release.set()
            first.result(timeout=1)
            self.assertEqual(executor.submit([3]).result(timeout=1), {"value": 3})
            self.assertTrue(executor.ready)
        finally:
            model.release.set()
            executor.close()

    def test_shutdown_callbacks_can_submit_and_close_without_deadlock(self):
        from inference_service.batching import ExecutorClosed
        model = BlockingModel()
        executor = BatchExecutor(model, Metrics(), 1, 0, 8)
        executor.submit([1])
        self.assertTrue(model.started.wait(1))
        waiting = executor.submit([2])
        observed = threading.Event()
        errors = []

        def callback(_future):
            try:
                with self.assertRaises(ExecutorClosed):
                    executor.submit([3])
                model.release.set()
                executor.close()
                observed.set()
            except Exception as error:
                errors.append(error)

        waiting.add_done_callback(callback)
        closer = threading.Thread(target=executor.close, daemon=True)
        closer.start()
        try:
            self.assertTrue(observed.wait(2), errors)
            closer.join(timeout=1)
            self.assertFalse(closer.is_alive())
            self.assertEqual(errors, [])
        finally:
            model.release.set()

    def test_worker_completion_callback_can_close_its_own_executor(self):
        model = BlockingModel()
        executor = BatchExecutor(model, Metrics(), 1, 0, 8)
        first = executor.submit([1])
        self.assertTrue(model.started.wait(1))
        completed = threading.Event()
        errors = []

        def callback(_future):
            try:
                executor.close()
                completed.set()
            except Exception as error:
                errors.append(error)

        first.add_done_callback(callback)
        model.release.set()
        try:
            self.assertTrue(completed.wait(2), errors)
            self.assertFalse(executor.ready)
            self.assertEqual(errors, [])
        finally:
            executor.close()


if __name__ == "__main__":
    unittest.main()
