#!/usr/bin/env python3
"""Tiny concurrent load generator using only the Python standard library."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import statistics
import time
import urllib.error
import urllib.request


def send(url: str, number: int) -> tuple:
    body = json.dumps({"inputs": [number / 100, 0.25, -0.1]}).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "X-Request-ID": f"load-{number}"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()
            return response.status, time.perf_counter() - started
    except urllib.error.HTTPError as exc:
        return exc.code, time.perf_counter() - started
    except urllib.error.URLError:
        return "network_error", time.perf_counter() - started


def percentile(values: list, fraction: float) -> float:
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * fraction), len(ordered) - 1)
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Send concurrent requests to Batchline")
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1/infer")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()

    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(send, args.url, number) for number in range(args.requests)]
        for future in as_completed(futures):
            results.append(future.result())
    elapsed = time.perf_counter() - started
    latencies = [latency for _, latency in results]
    status_counts = {status: sum(1 for code, _ in results if code == status) for status, _ in results}

    print(f"requests:   {len(results)}")
    print(f"statuses:   {status_counts}")
    print(f"throughput: {len(results) / elapsed:.1f} requests/second")
    print(f"latency:    p50={statistics.median(latencies) * 1000:.1f} ms "
          f"p95={percentile(latencies, 0.95) * 1000:.1f} ms")


if __name__ == "__main__":
    main()
