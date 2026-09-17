"""HTTP API for the Batchline inference service."""

from concurrent.futures import TimeoutError as FutureTimeout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
import signal
import threading
import time
import uuid
from typing import Dict, Optional, Tuple, Type

from .batching import BatchExecutor, QueueFull
from .config import Config
from .metrics import Metrics
from .model import ToyModel


LOGGER = logging.getLogger("batchline")
MAX_BODY_BYTES = 1_000_000


class InferenceServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    # The stdlib default is only 5. Bursty inference traffic can fill that TCP
    # accept backlog before application-level backpressure gets a chance to act.
    request_queue_size = 128

    def __init__(
        self,
        address: Tuple[str, int],
        handler: Type[BaseHTTPRequestHandler],
        executor: BatchExecutor,
        metrics: Metrics,
        timeout_seconds: float,
    ) -> None:
        super().__init__(address, handler)
        self.executor = executor
        self.metrics = metrics
        self.timeout_seconds = timeout_seconds
        self.started_at = time.time()


class Handler(BaseHTTPRequestHandler):
    server: InferenceServer
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/healthz":
            self._json(200, {"status": "ok"})
        elif self.path == "/readyz":
            ready = self.server.executor.ready
            self._json(200 if ready else 503, {"ready": ready})
        elif self.path == "/metrics":
            body = self.server.metrics.render().encode()
            self._send(200, body, "text/plain; version=0.0.4; charset=utf-8")
        else:
            self._json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/v1/infer":
            self._json(404, {"error": "not_found"})
            return

        request_id = self.headers.get("X-Request-ID") or str(uuid.uuid4())
        try:
            payload = self._read_json()
            inputs = validate_inputs(payload)
            future = self.server.executor.submit(inputs)
            prediction = future.result(timeout=self.server.timeout_seconds)
            self._json(200, {"request_id": request_id, "prediction": prediction}, request_id)
        except ValueError as exc:
            self._json(422, {"error": "invalid_request", "message": str(exc)}, request_id)
        except QueueFull:
            self._json(429, {"error": "queue_full", "message": "retry later"}, request_id)
        except FutureTimeout:
            self._json(504, {"error": "inference_timeout"}, request_id)
        except Exception:
            LOGGER.exception("inference failed request_id=%s", request_id)
            self._json(500, {"error": "internal_error"}, request_id)

    def _read_json(self) -> object:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("Content-Length is required")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("Content-Length must be an integer") from exc
        if length < 0 or length > MAX_BODY_BYTES:
            raise ValueError(f"request body must be at most {MAX_BODY_BYTES} bytes")
        try:
            return json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("body must be valid JSON") from exc

    def _json(self, status: int, payload: Dict[str, object], request_id: Optional[str] = None) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self._send(status, body, "application/json", request_id)

    def _send(
        self, status: int, body: bytes, content_type: str, request_id: Optional[str] = None
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if request_id:
            self.send_header("X-Request-ID", request_id)
        self.end_headers()
        self.wfile.write(body)
        self.server.metrics.request(self.command, status)

    def log_message(self, format_string: str, *args: object) -> None:
        LOGGER.info("%s - %s", self.client_address[0], format_string % args)


def validate_inputs(payload: object) -> list:
    if not isinstance(payload, dict):
        raise ValueError("body must be a JSON object")
    inputs = payload.get("inputs")
    if not isinstance(inputs, list) or isinstance(inputs, (str, bytes)):
        raise ValueError("inputs must be an array of numbers")
    if not 1 <= len(inputs) <= 256:
        raise ValueError("inputs must contain between 1 and 256 numbers")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in inputs):
        raise ValueError("every input must be a number")
    converted = [float(value) for value in inputs]
    if not all(math.isfinite(value) for value in converted):
        raise ValueError("inputs must be finite numbers")
    return converted


def build_server(config: Config) -> InferenceServer:
    metrics = Metrics()
    model = ToyModel(latency_ms=config.model_latency_ms)
    executor = BatchExecutor(
        model=model,
        metrics=metrics,
        max_batch_size=config.max_batch_size,
        batch_window_ms=config.batch_window_ms,
        max_queue_size=config.max_queue_size,
    )
    try:
        return InferenceServer(
            (config.host, config.port),
            Handler,
            executor,
            metrics,
            config.request_timeout_ms / 1000,
        )
    except Exception:
        executor.close()
        raise


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = Config.from_env()
    server = build_server(config)

    def shutdown(_signum: int, _frame: object) -> None:
        # shutdown() must run from a different thread than serve_forever().
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    LOGGER.info("listening on http://%s:%s", config.host, server.server_port)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        server.executor.close()
        LOGGER.info("stopped")


if __name__ == "__main__":
    main()
