# Batchline

Batchline is a small inference-serving system built to teach the infrastructure around a model. It is deliberately dependency-free: the model is tiny, while the queueing, batching, overload protection, observability, lifecycle, container, tests, and load generator are real.

```text
HTTP clients ──► threaded API ──► bounded queue ──► batch worker ──► model
                       │                │                 │
                       └── status ◄─────┴──── metrics ────┘
```

## What you will learn

- **Serving:** expose inference through a versioned HTTP endpoint.
- **Dynamic batching:** collect nearby requests and execute them together.
- **Backpressure:** cap the queue and reject excess work with HTTP 429.
- **Timeouts:** give callers a bounded wait instead of hanging forever.
- **Observability:** export request, queue, batch, and latency metrics.
- **Operations:** health/readiness checks, graceful shutdown, containers, and CI.

## Run it

Requires Python 3.9 or newer and no third-party packages.

```bash
make test
make run
```

In another terminal:

```bash
curl -s http://localhost:8080/v1/infer \
  -H 'Content-Type: application/json' \
  -d '{"inputs":[0.2,1.5,-0.3]}'

curl -s http://localhost:8080/metrics
make load
```

Or run it in a container:

```bash
docker compose up --build
```

## API

`POST /v1/infer`

```json
{
  "inputs": [0.2, 1.5, -0.3]
}
```

Successful response:

```json
{
  "request_id": "66bbc7fb-17c2-4568-9e4f-44fa7d692cab",
  "prediction": {
    "label": "positive",
    "score": 0.71095,
    "model_version": "toy-logistic-v1"
  }
}
```

Other endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /healthz` | Process is alive |
| `GET /readyz` | Batch worker can accept work |
| `GET /metrics` | Prometheus-format metrics |

## Configuration

| Environment variable | Default | Meaning |
|---|---:|---|
| `HOST` | `0.0.0.0` | Listen address |
| `PORT` | `8080` | Listen port |
| `MAX_BATCH_SIZE` | `8` | Largest model batch |
| `BATCH_WINDOW_MS` | `10` | Maximum time spent gathering a batch |
| `MAX_QUEUE_SIZE` | `128` | Waiting requests before overload rejection |
| `REQUEST_TIMEOUT_MS` | `2000` | Caller wait limit |
| `MODEL_LATENCY_MS` | `20` | Simulated accelerator work per batch |

## Suggested learning path

1. Run one request and inspect its response.
2. Run `make load`, then inspect `/metrics`; compare batches to batch items.
3. Change `MAX_BATCH_SIZE` and `BATCH_WINDOW_MS` and observe throughput versus latency.
4. Set `MAX_QUEUE_SIZE=1 MODEL_LATENCY_MS=500` and create overload; find the 429s.
5. Read [the learning guide](docs/LEARNING_GUIDE.md) alongside the source.
6. Replace `ToyModel` with a real CPU model, then add multiple workers or a GPU runtime.

This is an educational baseline, not a claim of production readiness. Authentication, TLS, distributed routing, model artifact storage, autoscaling, tracing, and a real accelerator runtime are intentionally left as next steps.
