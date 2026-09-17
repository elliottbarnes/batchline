"""Environment-based service configuration."""

from dataclasses import dataclass
import os


def _integer(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True)
class Config:
    host: str
    port: int
    max_batch_size: int
    batch_window_ms: int
    max_queue_size: int
    request_timeout_ms: int
    model_latency_ms: int

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            host=os.getenv("HOST", "0.0.0.0"),
            port=_integer("PORT", 8080, 0, 65535),
            max_batch_size=_integer("MAX_BATCH_SIZE", 8, 1, 1024),
            batch_window_ms=_integer("BATCH_WINDOW_MS", 10, 0, 1000),
            max_queue_size=_integer("MAX_QUEUE_SIZE", 128, 1, 100_000),
            request_timeout_ms=_integer("REQUEST_TIMEOUT_MS", 2000, 1, 300_000),
            model_latency_ms=_integer("MODEL_LATENCY_MS", 20, 0, 60_000),
        )
