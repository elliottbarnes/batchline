import json
import threading
import unittest
import urllib.error
import urllib.request

from inference_service.config import Config
from inference_service.server import build_server, validate_inputs


class ValidationTests(unittest.TestCase):
    def test_accepts_numeric_array(self) -> None:
        self.assertEqual(validate_inputs({"inputs": [1, 2.5]}), [1.0, 2.5])

    def test_rejects_booleans_and_non_finite_values(self) -> None:
        for value in (True, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_inputs({"inputs": [value]})


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = Config(
            host="127.0.0.1",
            port=0,
            max_batch_size=4,
            batch_window_ms=2,
            max_queue_size=8,
            request_timeout_ms=1000,
            model_latency_ms=0,
        )
        cls.server = build_server(config)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.server.executor.close()
        cls.thread.join(timeout=1)

    def test_health_and_inference(self) -> None:
        with urllib.request.urlopen(self.base_url + "/healthz") as response:
            self.assertEqual(json.load(response), {"status": "ok"})

        request = urllib.request.Request(
            self.base_url + "/v1/infer",
            data=json.dumps({"inputs": [1, 2, 3]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as response:
            payload = json.load(response)
        self.assertEqual(payload["prediction"]["label"], "positive")
        self.assertIn("request_id", payload)

    def test_invalid_request_returns_422(self) -> None:
        request = urllib.request.Request(
            self.base_url + "/v1/infer",
            data=json.dumps({"inputs": []}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        self.assertEqual(error.exception.code, 422)

    def test_metrics_are_exposed(self) -> None:
        with urllib.request.urlopen(self.base_url + "/metrics") as response:
            metrics = response.read().decode()
        self.assertIn("inference_http_requests_total", metrics)
        self.assertIn("inference_batches_total", metrics)

    def test_listener_backlog_handles_bursty_traffic(self) -> None:
        self.assertGreaterEqual(self.server.request_queue_size, 128)


if __name__ == "__main__":
    unittest.main()
