# Learning guide: inference infrastructure from first principles

## 1. The system you built

An inference system turns a model artifact into a reliable online service. Batchline separates that work into a request path and an execution path:

```text
request                    shared infrastructure                     response
   │                                                                  ▲
   ├─ validate ─► bounded queue ─► gather for ≤ batch window           │
   │                                  │                               │
   │                                  └─► model(batch) ─► split results┘
   │
   └─ errors: 422 bad input · 429 overload · 504 deadline
```

The important boundary is the queue. HTTP threads can receive many requests, but the model worker controls expensive execution. This is the simplest form of admission control and scheduling.

There are actually two queues: the operating system's TCP accept backlog and the application's inference queue. Batchline raises the small standard-library backlog so bursty connections reach the application, where explicit queue limits can return a meaningful 429 response.

## 2. Why batching works

Accelerators and many CPU kernels process a matrix more efficiently than a single row. Model execution also has fixed overhead: scheduling, memory copies, and kernel launches. If one batch costs roughly 20 ms whether it contains one request or eight, then:

```text
batch size 1:  1 / 0.020 =  50 requests/second
batch size 8:  8 / 0.020 = 400 requests/second (idealized)
```

The cost is latency. The first item may wait for `BATCH_WINDOW_MS` while the worker gathers companions. Real systems tune the window against a service-level objective (SLO), such as “99% of requests finish within 100 ms.”

Read `BatchExecutor._work` in `src/inference_service/batching.py`. Notice the two batch-ending conditions: size limit and time limit.

## 3. Backpressure is a feature

An unbounded queue does not create capacity. During overload it only converts a clear failure into rising memory use and very slow responses. Batchline limits the queue and returns 429 when it is full. A caller can retry with exponential backoff and jitter.

In a larger deployment, backpressure appears at several layers:

- load balancer connection limits;
- per-tenant rate limits;
- server queue limits;
- accelerator memory limits;
- autoscaler maximums.

## 4. Health is not readiness

`/healthz` says the HTTP process is alive. `/readyz` says the model worker is running and the instance is suitable for traffic. Orchestrators use these differently: liveness may restart a broken process, while readiness removes it from routing without necessarily restarting it.

A production readiness check would also confirm that model weights are loaded and perhaps that a small warmup inference succeeded.

## 5. What to measure

Batchline exposes Prometheus text metrics without requiring a Prometheus server:

- request counts by status show errors and overload;
- queue depth is an early signal of saturation;
- batches versus batch items reveals achieved batch size;
- inference duration shows model execution cost.

Useful derived signals include:

```text
average batch size = rate(batch_items_total) / rate(batches_total)
error ratio        = rate(requests{status!="200"}) / rate(requests)
queue saturation   = queue_depth / max_queue_size
```

Avoid putting request IDs or user IDs in metric labels: every unique label combination creates a time series and can overwhelm the metrics backend. Request IDs belong in logs and traces.

## 6. Model and service lifecycle

The container starts one process and one batch worker. `SIGTERM` stops accepting traffic, shuts down the HTTP loop, and then closes the worker. This is the beginning of graceful shutdown.

A production implementation would add a drain period so already accepted requests finish before the orchestrator terminates the container. It would also warm the model before advertising readiness.

## 7. Experiments

### Throughput versus latency

Run the service, then change one variable at a time:

```bash
MAX_BATCH_SIZE=1 make run
MAX_BATCH_SIZE=16 BATCH_WINDOW_MS=25 make run
```

For each setting, run `make load`. Record throughput, p50, p95, and average batch size. You should see that a larger batching window can increase achieved batch size under moderate load but also adds waiting time.

### Force overload

```bash
MAX_QUEUE_SIZE=1 MAX_BATCH_SIZE=1 MODEL_LATENCY_MS=500 make run
python3 scripts/load_test.py --requests 30 --concurrency 30
```

The 429 responses are expected. The service is protecting itself rather than accepting work it cannot finish promptly.

### Break the worker safely

Modify the model to raise an exception for a chosen value. Verify that callers receive 500 but the worker stays alive for later requests. Then add a dedicated model-error counter.

## 8. Where real systems go next

The concepts stay the same when the toy model becomes a large language model, but the constraints get sharper:

- **Model runtime:** ONNX Runtime, PyTorch, TensorRT-LLM, vLLM, or another optimized engine.
- **Scheduling:** continuous batching for token generation instead of one batch per request.
- **Memory:** weight placement, KV-cache allocation, quantization, and fragmentation.
- **Routing:** model-aware and sometimes cache-aware routing across replicas.
- **Autoscaling:** use queue delay or pending requests, not CPU alone, for accelerator workloads.
- **Artifacts:** versioned model registry, integrity checks, rollout, and rollback.
- **Observability:** traces spanning gateway, scheduler, model execution, and streaming output.

The clean seam to extend is the model object: keep `predict_batch(batch)` and swap its implementation. The next architectural leap is multiple worker processes or machines, which requires routing and distributed coordination rather than just more threads.
