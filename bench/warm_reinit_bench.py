"""Measure warm reinit on a live SGLang server.

Three reinits in a row, each followed by a real generation to prove the
rebuilt pools and graphs serve. Records the engine's own timings (free and
rebuild) and the wall time from the control call to the first token after.
Compare with boot_patched_s from pod_run.sh, which is what a relaunch costs.

    python3 bench/warm_reinit_bench.py --base http://127.0.0.1:30000 --out warm.json
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trimtab.adapters import make_adapter  # noqa: E402


def generate(base, engine):
    if engine == "sglang":
        req = urllib.request.Request(base + "/generate", data=json.dumps(
            {"text": "The ship turned when", "sampling_params": {"max_new_tokens": 16, "temperature": 0}}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["text"]
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(
        {"model": "default", "prompt": "The ship turned when", "max_tokens": 16, "temperature": 0}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["choices"][0]["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--engine", default="sglang", choices=["sglang", "vllm"])
    ap.add_argument("--out", default="warm_reinit.json")
    a = ap.parse_args()
    ad = make_adapter(a.engine, a.base)
    raw = ad.read_raw()
    if a.engine == "sglang":
        boot_tokens, boot_mrr = raw["max_total_num_tokens"], raw["max_running_requests"]
        plan = [
            {"mem_fraction_static": 0.75},
            {"max_total_tokens": max(4096, boot_tokens // 4), "max_running_requests": max(1, boot_mrr // 2)},
            {"mem_fraction_static": 0.85},
        ]
        size_key = "max_total_num_tokens"
    else:
        boot_tokens, boot_mrr = raw.get("num_gpu_blocks"), raw["max_num_seqs"]
        plan = [
            {"gpu_memory_utilization": 0.7},
            {"gpu_memory_utilization": 0.6, "max_num_seqs": max(1, boot_mrr // 2)},
            {"gpu_memory_utilization": 0.85, "max_num_seqs": boot_mrr},
        ]
        size_key = "num_gpu_blocks"
    print(f"boot {size_key}={boot_tokens} cap={boot_mrr}")
    rows = []
    for fields in plan:
        t0 = time.perf_counter()
        r = ad.reinit_warm(fields)
        call_s = time.perf_counter() - t0
        text, gen_ok = "", False
        try:
            text = generate(a.base, a.engine); gen_ok = bool(text)
        except Exception as e:
            text = f"generate failed: {e}"
        first_token_s = time.perf_counter() - t0
        after = ad.read_raw()
        row = dict(fields=fields, ok=r.ok, http=r.http_status, engine=r.applied.get("last_reinit"),
                   call_s=round(call_s, 2), call_to_first_token_s=round(first_token_s, 2),
                   max_total_num_tokens_after=after.get(size_key), max_running_requests_after=after.get("max_running_requests", after.get("max_num_seqs")),
                   generate_ok=gen_ok, sample=text[:60])
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "sample"}))
        if not (r.ok and gen_ok):
            break
    summary = dict(boot_max_total_num_tokens=boot_tokens, boot_max_running_requests=boot_mrr,
                   all_ok=all(x["ok"] and x["generate_ok"] for x in rows) and len(rows) == len(plan), rows=rows)
    json.dump(summary, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}))
    sys.exit(0 if summary["all_ok"] else 1)


if __name__ == "__main__":
    main()
