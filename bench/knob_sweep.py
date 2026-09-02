"""Set every hot knob in the manifest on a live engine, read it back, restore.

One row per knob with the HTTP status, so the results say which knobs were
proven on which engine version and GPU, not just the concurrency cap.

    python3 bench/knob_sweep.py --engine sglang --base http://127.0.0.1:30000 --out sweep.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trimtab import manifest  # noqa: E402
from trimtab.adapters import make_adapter  # noqa: E402

SAMPLES = {
    "max_running_requests": lambda boot: max(1, boot // 2),
    "max_queued_requests": lambda boot: 512,
    "chunked_prefill_size": lambda boot: 2048,
    "max_prefill_tokens": lambda boot: 4096,
    "schedule_policy": lambda boot: "lpm",
    "schedule_conservativeness": lambda boot: 0.5,
    "log_level": lambda boot: "WARNING",
    "max_num_seqs": lambda boot: max(1, boot // 2),
    "max_num_batched_tokens": lambda boot: max(1, boot // 2),
    "long_prefill_token_threshold": lambda boot: 2048,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", default="knob_sweep.json")
    a = ap.parse_args()
    adapter = make_adapter(a.engine, a.base)
    m = manifest.find(a.engine)
    before = adapter.read_knobs()
    rows = []
    for k in m.hot_fields():
        if k not in SAMPLES or k not in before:
            continue
        target = SAMPLES[k](before[k] if isinstance(before[k], int) else 0)
        r = adapter.set_hot({k: target})
        after = adapter.read_knobs().get(k)
        ok = r.ok and str(after).lower() == str(target).lower()
        restore = adapter.set_hot({k: before[k]}) if before[k] is not None else None
        rows.append(dict(knob=k, boot=before[k], target=target, read_back=after, http=r.http_status,
                         api_ms=round(r.latency_ms, 2), ok=ok, restored=bool(restore and restore.ok)))
        print(f"{k:<28} boot={before[k]!s:<8} set={target!s:<8} read={after!s:<8} http={r.http_status} ok={ok}")
    summary = dict(engine=a.engine, knobs_ok=sum(r["ok"] for r in rows), knobs_total=len(rows), rows=rows)
    json.dump(summary, open(a.out, "w"), indent=1)
    sys.exit(0 if summary["knobs_ok"] == summary["knobs_total"] else 1)


if __name__ == "__main__":
    main()
