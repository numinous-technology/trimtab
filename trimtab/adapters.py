"""Engine adapters. One protocol, one result shape, two engines.

An adapter turns a dict of hot knob changes into the engine's control call and
reads the live values back. Validation of values against the manifest happens
before the adapter is called (boundary discipline). The adapter reports what
the engine itself accepted, since the engine holds the physical ceilings.
"""
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass
class ApplyResult:
    ok: bool
    applied: dict = field(default_factory=dict)
    rejected: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    http_status: int = 0


def _post(url, payload, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def _get(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def _all_updated(body):
    """SGLang fans the request out to every rank and returns one entry per
    rank, either a bare bool or {"updated": bool}. Every rank must agree."""
    entries = body if isinstance(body, list) else [body]
    if not entries:
        return False
    return all(e if isinstance(e, bool) else bool(e.get("updated")) for e in entries)


class SGLangAdapter:
    """POST /set_internal_state, read back through /get_server_info."""

    name = "sglang"

    def __init__(self, base_url, timeout=30):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def set_hot(self, changes: dict) -> ApplyResult:
        t0 = time.perf_counter()
        status, body = _post(f"{self.base}/set_internal_state", {"server_args": changes}, self.timeout)
        ms = (time.perf_counter() - t0) * 1000
        ok = status == 200 and _all_updated(body)
        live = self.read_knobs() if ok else {}
        applied = {k: v for k, v in changes.items() if live.get(k) == v}
        rejected = {} if ok else {k: "engine rejected the update" for k in changes}
        return ApplyResult(ok and applied.keys() == changes.keys(), applied, rejected, ms, status)

    WARM = ("mem_fraction_static", "max_total_tokens", "kv_cache_dtype", "max_running_requests")

    def reinit_warm(self, fields: dict, timeout=900) -> ApplyResult:
        """Rebuild pools, backends and graphs in place. Weights stay on the GPU.
        The engine refuses unless idle, so drain first. Returns the engine's
        timing in applied["last_reinit"]."""
        bad = sorted(set(fields) - set(self.WARM))
        if bad:
            return ApplyResult(False, rejected={k: "not warm-reinitable" for k in bad})
        t0 = time.perf_counter()
        status, body = _post(f"{self.base}/set_internal_state", {"server_args": {f"reinit.{k}": v for k, v in fields.items()}}, timeout)
        ms = (time.perf_counter() - t0) * 1000
        ok = status == 200 and _all_updated(body)
        last = self.read_raw().get("last_reinit") if ok else None
        return ApplyResult(ok, {"last_reinit": last} if ok else {}, {} if ok else {k: "engine refused, see server log" for k in fields}, ms, status)

    def read_raw(self) -> dict:
        info = _get(f"{self.base}/get_server_info", self.timeout)
        states = info.get("internal_states") or []
        return states[0].get("trimtab", {}) if states else {}

    def read_knobs(self) -> dict:
        info = _get(f"{self.base}/get_server_info", self.timeout)
        states = info.get("internal_states") or []
        if states and "trimtab" in states[0]:
            return {k: v for k, v in states[0]["trimtab"].items() if k not in ("ceilings", "last_reinit", "max_total_num_tokens")}
        if states:
            return {"max_running_requests": states[0].get("effective_max_running_requests_per_dp")}
        return {}

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base}/health", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False


class VLLMAdapter:
    """POST /trimtab/set_knobs, read back through GET /trimtab/knobs.
    Requires the server to run with VLLM_SERVER_DEV_MODE=1."""

    name = "vllm"

    def __init__(self, base_url, timeout=30):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def set_hot(self, changes: dict) -> ApplyResult:
        t0 = time.perf_counter()
        status, body = _post(f"{self.base}/trimtab/set_knobs", changes, self.timeout)
        ms = (time.perf_counter() - t0) * 1000
        return ApplyResult(bool(body.get("ok")), body.get("applied", {}), body.get("rejected", {}), ms, status)

    def read_knobs(self) -> dict:
        body = _get(f"{self.base}/trimtab/knobs", self.timeout)
        return {k: v for k, v in body.items() if k not in ("ceilings", "running")}

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base}/health", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False


ADAPTERS = {"sglang": SGLangAdapter, "vllm": VLLMAdapter}


def make_adapter(engine: str, base_url: str):
    try:
        return ADAPTERS[engine](base_url)
    except KeyError:
        raise ValueError(f"unknown engine {engine!r}, known: {sorted(ADAPTERS)}")
