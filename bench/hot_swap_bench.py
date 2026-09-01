"""Measure trimtab hot-swap latency on a live engine under load.

Runs background generation load, then flips the running-request cap between
a low value and its boot value, measuring per change the API latency (request
sent to engine confirming on every rank) and time to effect (value read back
from the scheduler equals the target). Background request failures are
counted across every swap for the zero-drop claim.

Every row records the HTTP status of the control call, so a fast failure is
never mistaken for a fast success.

    python3 hot_swap_bench.py --engine sglang --base http://127.0.0.1:30000 --iters 20
    python3 hot_swap_bench.py --engine vllm   --base http://127.0.0.1:8000  --iters 20
"""
import argparse
import concurrent.futures as cf
import json
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trimtab.adapters import make_adapter  # noqa: E402

CAP = {"sglang": "max_running_requests", "vllm": "max_num_seqs"}
GEN = {
    "sglang": ("/generate", lambda n: {"text": "Write a paragraph about ships. ",
                                       "sampling_params": {"max_new_tokens": n, "temperature": 0.7}}),
    "vllm": ("/v1/completions", lambda n: {"model": "default", "prompt": "Write a paragraph about ships. ",
                                           "max_tokens": n, "temperature": 0.7}),
}


class Load:
    def __init__(self, base, path, payload, concurrency):
        self.base, self.path, self.payload, self.n = base, path, payload, concurrency
        self.stop, self.lock, self.sent, self.failed = threading.Event(), threading.Lock(), 0, 0

    def worker(self):
        req = urllib.request.Request(self.base + self.path, data=json.dumps(self.payload).encode(),
                                     headers={"Content-Type": "application/json"})
        while not self.stop.is_set():
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    r.read()
                with self.lock:
                    self.sent += 1
            except Exception:
                with self.lock:
                    self.failed += 1

    def start(self):
        self.pool = cf.ThreadPoolExecutor(max_workers=self.n)
        for _ in range(self.n):
            self.pool.submit(self.worker)

    def halt(self):
        self.stop.set()
        self.pool.shutdown(wait=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=sorted(CAP))
    ap.add_argument("--base", required=True)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--load-concurrency", type=int, default=32)
    ap.add_argument("--load-max-tokens", type=int, default=128)
    ap.add_argument("--low", type=int, default=8)
    ap.add_argument("--settle-s", type=float, default=10)
    ap.add_argument("--gap-s", type=float, default=2)
    ap.add_argument("--vllm-model", default="default")
    ap.add_argument("--out", default="hot_swap_results.json")
    a = ap.parse_args()

    adapter = make_adapter(a.engine, a.base)
    cap = CAP[a.engine]
    boot = adapter.read_knobs()[cap]
    print(f"engine={a.engine} boot {cap}={boot}")

    path, mk = GEN[a.engine]
    payload = mk(a.load_max_tokens)
    if a.engine == "vllm":
        payload["model"] = a.vllm_model
    load = Load(a.base, path, payload, a.load_concurrency)
    load.start()
    time.sleep(a.settle_s)

    rows = []
    for i in range(a.iters):
        target = a.low if i % 2 == 0 else boot
        t0 = time.perf_counter()
        r = adapter.set_hot({cap: target})
        eff = None
        while eff != target and time.perf_counter() - t0 < 10:
            eff = adapter.read_knobs()[cap]
        effect_ms = (time.perf_counter() - t0) * 1000
        ok = r.ok and eff == target
        rows.append(dict(target=target, api_ms=r.latency_ms, effect_ms=effect_ms, http=r.http_status, ok=ok))
        print(f"iter {i:>2} target={target:<5} api={r.latency_ms:7.1f}ms effect={effect_ms:7.1f}ms http={r.http_status} ok={ok}")
        time.sleep(a.gap_s)

    time.sleep(3)
    load.halt()

    api = sorted(x["api_ms"] for x in rows)
    eff = sorted(x["effect_ms"] for x in rows)
    p95 = lambda v: v[max(0, int(len(v) * 0.95) - 1)]
    summary = dict(
        engine=a.engine, cap_knob=cap, boot_value=boot, iters=len(rows), all_ok=all(x["ok"] for x in rows),
        api_ms_median=round(statistics.median(api), 2), api_ms_p95=round(p95(api), 2),
        effect_ms_median=round(statistics.median(eff), 2), effect_ms_p95=round(p95(eff), 2),
        background_completed=load.sent, background_failed=load.failed, rows=rows,
    )
    json.dump(summary, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
    sys.exit(0 if summary["all_ok"] else 1)


if __name__ == "__main__":
    main()
