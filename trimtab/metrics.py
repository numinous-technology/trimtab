"""Prometheus text scraper that turns engine /metrics into gate inputs.

Both SGLang (with --enable-metrics) and vLLM expose Prometheus text. Gauges
pass through by name. Counters become per-second rates between this sample
and the previous one for the same replica. Histograms become p50, p95, p99
estimated from the bucket deltas between samples, which is the same
interpolation Prometheus itself uses for histogram_quantile.

Metric names differ per engine, so a name map turns engine names into the
gate vocabulary (error_rate, p99_ttft_ms, throughput). Missing metrics stay
missing rather than defaulting to zero, so a gate on an absent metric fails
loudly instead of passing silently.
"""
import re
import time
import urllib.request

LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE+naNIif]+)')

SGLANG_MAP = {
    "ttft_s": "sglang:time_to_first_token_seconds",
    "e2e_s": "sglang:e2e_request_latency_seconds",
    "gen_tokens": "sglang:generation_tokens_total",
    "running": "sglang:num_running_reqs",
    "queued": "sglang:num_queue_reqs",
}
VLLM_MAP = {
    "ttft_s": "vllm:time_to_first_token_seconds",
    "e2e_s": "vllm:e2e_request_latency_seconds",
    "gen_tokens": "vllm:generation_tokens_total",
    "running": "vllm:num_requests_running",
    "queued": "vllm:num_requests_waiting",
}


def parse(text):
    """-> {name: {"value": float} | {"buckets": {le: cum}, "count": c, "sum": s}}."""
    out = {}
    for raw in text.splitlines():
        if not raw or raw[0] == "#":
            continue
        m = LINE.match(raw)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", float(m.group(3))
        if name.endswith("_bucket"):
            le = re.search(r'le="([^"]+)"', labels)
            base = name[:-7]
            out.setdefault(base, {"buckets": {}, "count": 0.0, "sum": 0.0})["buckets"][float(le.group(1)) if le.group(1) != "+Inf" else float("inf")] = \
                out.setdefault(base, {"buckets": {}, "count": 0.0, "sum": 0.0})["buckets"].get(float(le.group(1)) if le.group(1) != "+Inf" else float("inf"), 0.0) + val
        elif name.endswith("_count") and name[:-6] in out:
            out[name[:-6]]["count"] += val
        elif name.endswith("_sum") and name[:-4] in out:
            out[name[:-4]]["sum"] += val
        else:
            out.setdefault(name, {"value": 0.0})
            out[name]["value"] = out[name].get("value", 0.0) + val
    return out


def quantile(q, buckets_delta):
    """Prometheus histogram_quantile over bucket deltas {le: count_in_window}."""
    les = sorted(buckets_delta)
    total = buckets_delta[les[-1]] if les else 0
    if total <= 0:
        return None
    rank, cum, prev_le, prev_cum = q * total, 0.0, 0.0, 0.0
    for le in les:
        cum = buckets_delta[le]
        if cum >= rank:
            if le == float("inf"):
                return prev_le
            if cum == prev_cum:
                return le
            return prev_le + (le - prev_le) * (rank - prev_cum) / (cum - prev_cum)
        prev_le, prev_cum = le, cum
    return les[-1]


class PrometheusMetrics:
    def __init__(self, urls: dict, name_map: dict, fetch=None):
        self.urls, self.map, self.prev = urls, name_map, {}
        self.fetch = fetch or (lambda u: urllib.request.urlopen(u, timeout=10).read().decode())

    def sample(self, replica_id) -> dict:
        now, cur = time.monotonic(), parse(self.fetch(self.urls[replica_id]))
        out = {}
        for k in ("running", "queued"):
            n = self.map.get(k)
            if n in cur and "value" in cur[n]:
                out[k] = cur[n]["value"]
        prev = self.prev.get(replica_id)
        if prev:
            pt, p = prev
            dt = max(now - pt, 1e-6)
            g = self.map.get("gen_tokens")
            if g in cur and g in p:
                out["throughput"] = (cur[g]["value"] - p[g]["value"]) / dt
            for key, label in (("ttft_s", "ttft"), ("e2e_s", "e2e")):
                h = self.map.get(key)
                if h in cur and h in p and "buckets" in cur[h]:
                    delta = {le: cur[h]["buckets"][le] - p[h]["buckets"].get(le, 0.0) for le in cur[h]["buckets"]}
                    for q, nm in ((0.5, "p50"), (0.95, "p95"), (0.99, "p99")):
                        v = quantile(q, delta)
                        if v is not None:
                            out[f"{nm}_{label}_ms"] = v * 1000
                    n = cur[h]["count"] - p[h]["count"]
                    if n > 0:
                        out[f"{label}_requests"] = n
        self.prev[replica_id] = (now, cur)
        return out
