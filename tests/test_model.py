import math
import unittest

from inference_service.model import ToyModel


class ToyModelTests(unittest.TestCase):
    def test_predicts_one_result_per_input(self) -> None:
        model = ToyModel(latency_ms=0)
        results = model.predict_batch([[1.0, 2.0], [-1.0, -2.0]])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["label"], "positive")
        self.assertEqual(results[1]["label"], "negative")

    def test_sigmoid_stays_stable_for_large_values(self) -> None:
        model = ToyModel(latency_ms=0)
        results = model.predict_batch([[1e308], [-1e308]])
        self.assertTrue(math.isfinite(results[0]["score"]))
        self.assertTrue(math.isfinite(results[1]["score"]))


if __name__ == "__main__":
    unittest.main()
