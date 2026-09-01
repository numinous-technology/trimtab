"""Measure trimtab hot-swap latency on a live SGLang server under load.

Runs background generation load, then repeatedly changes hot knobs through
POST /set_internal_state and measures two things per change. The API latency,
from request sent to the engine confirming the change on every rank. And the
time to effect for max_running_requests, confirmed by reading the value back
from the scheduler through /get_server_info.

Also counts background request failures across every swap, since the zero-drop
claim matters as much as the latency claim.

Usage
    python3 hot_swap_bench.py --base http://127.0.0.1:30000 --iters 20
"""
import argparse
import concurrent.futures as cf
import json
import statistics
import threading
import time
import urllib.request


def post(base, path, payload, timeout=120):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(base, path, timeout=30):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.loads(r.read())


def find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = find_key(v, key)
            if found is not None:
                return found
    if isinstance(obj, list):
        for v in obj:
            found = find_key(v, key)
            if found is not None:
                return found
    return None


def read_effective_mrr(base):
    return find_key(get(base, "/get_server_info"), "effective_max_running_requests_per_dp")


class Load:
    """Continuous background generation load."""

    def __init__(self, base, concurrency, max_tokens):
        self.base = base
        self.concurrency = concurrency
        self.max_tokens = max_tokens
        self.stop = threading.Event()
        self.sent = 0
        self.failed = 0
        self.lock = threading.Lock()

    def worker(self):
        while not self.stop.is_set():
            try:
                post(
                    self.base,
                    "/generate",
                    {
                        "text": "Write a paragraph about ships. ",
                        "sampling_params": {
                            "max_new_tokens": self.max_tokens,
                            "temperature": 0.7,
                        },
                    },
                )
                with self.lock:
                    self.sent += 1
            except Exception:
                with self.lock:
                    self.failed += 1

    def run(self):
        self.pool = cf.ThreadPoolExecutor(max_workers=self.concurrency)
        for _ in range(self.concurrency):
            self.pool.submit(self.worker)

    def halt(self):
        self.stop.set()
        self.pool.shutdown(wait=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:30000")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--load-concurrency", type=int, default=32)
    ap.add_argument("--load-max-tokens", type=int, default=128)
    ap.add_argument("--low", type=int, default=8)
    ap.add_argument("--out", default="hot_swap_results.json")
    a = ap.parse_args()

    boot_mrr = read_effective_mrr(a.base)
    print(f"boot max_running_requests={boot_mrr}")

    load = Load(a.base, a.load_concurrency, a.load_max_tokens)
    load.run()
    time.sleep(10)  # let load reach steady state

    rows = []
    for i in range(a.iters):
        target = a.low if i % 2 == 0 else boot_mrr
        t0 = time.perf_counter()
        resp = post(
            a.base,
            "/set_internal_state",
            {"server_args": {"max_running_requests": target}},
        )
        api_ms = (time.perf_counter() - t0) * 1000
        eff = None
        while eff != target and time.perf_counter() - t0 < 10:
            eff = read_effective_mrr(a.base)
        effect_ms = (time.perf_counter() - t0) * 1000
        ok = bool(find_key(resp, "updated")) and eff == target
        rows.append(dict(target=target, api_ms=api_ms, effect_ms=effect_ms, ok=ok))
        print(f"iter {i} target={target} api={api_ms:.1f}ms effect={effect_ms:.1f}ms ok={ok}")
        time.sleep(2)

    # one call each for the other two knobs, api latency only
    extra = {}
    for knob, val in [("max_queued_requests", 512), ("chunked_prefill_size", 2048)]:
        t0 = time.perf_counter()
        resp = post(a.base, "/set_internal_state", {"server_args": {knob: val}})
        extra[knob] = dict(
            api_ms=(time.perf_counter() - t0) * 1000,
            updated=bool(find_key(resp, "updated")),
        )
        print(f"{knob}={val} api={extra[knob]['api_ms']:.1f}ms updated={extra[knob]['updated']}")

    time.sleep(5)
    load.halt()

    api = sorted(r["api_ms"] for r in rows)
    eff = sorted(r["effect_ms"] for r in rows)
    summary = dict(
        boot_max_running_requests=boot_mrr,
        iters=len(rows),
        all_ok=all(r["ok"] for r in rows),
        api_ms_median=statistics.median(api),
        api_ms_p95=api[int(len(api) * 0.95) - 1],
        effect_ms_median=statistics.median(eff),
        effect_ms_p95=eff[int(len(eff) * 0.95) - 1],
        background_completed=load.sent,
        background_failed=load.failed,
        other_knobs=extra,
        rows=rows,
    )
    json.dump(summary, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
