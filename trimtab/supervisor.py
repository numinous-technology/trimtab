"""trimtabd. Owns one engine process and converges it on the desired version.

Hot fields go through the adapter while the engine runs. Cold fields change
the launch command, so the supervisor drains, stops the engine, relaunches
with the new flags, waits for health, and records how long that took. Weights
stay on local disk and warm in the page cache between launches, which is the
part of a redeploy this path removes (no image pull, no weight download).
That is the pinned-host path from the spec. GPU-resident reinit needs an
in-process path neither engine exposes yet and is not claimed here.

Cold fields map to flags through one table per engine. A cold field that is
not in the table is refused rather than guessed.
"""
import subprocess
import sys
import time
import urllib.request

from .adapters import make_adapter
from .manifest import COLD, Manifest
from .store import Store

COLD_FLAGS = {
    "sglang": {
        "tp_size": "--tp", "ep_size": "--ep-size", "quantization": "--quantization",
        "kv_cache_dtype": "--kv-cache-dtype", "mem_fraction_static": "--mem-fraction-static",
        "max_total_num_tokens": "--max-total-tokens", "cuda_graph_max_bs": "--cuda-graph-max-bs",
        "attention_backend": "--attention-backend", "page_size": "--page-size",
        "speculative_algorithm": "--speculative-algorithm",
        "speculative_draft_model_path": "--speculative-draft-model-path",
    },
    "vllm": {
        "tensor_parallel_size": "--tensor-parallel-size", "quantization": "--quantization",
        "kv_cache_dtype": "--kv-cache-dtype", "gpu_memory_utilization": "--gpu-memory-utilization",
        "max_model_len": "--max-model-len", "cuda_graph_sizes": "--cuda-graph-sizes",
        "speculative_config": "--speculative-config",
    },
    "mock": {"max_total_num_tokens": "--tokens"},
}


def base_command(engine, model, port, mock_engine="sglang"):
    if engine == "sglang":
        return ["python3", "-m", "sglang.launch_server", "--model-path", model, "--host", "0.0.0.0", "--port", str(port)]
    if engine == "vllm":
        return ["vllm", "serve", model, "--served-model-name", "default", "--host", "0.0.0.0", "--port", str(port)]
    if engine == "mock":
        return [sys.executable, "-m", "trimtab.mock_engine", "--engine", mock_engine, "--port", str(port)]
    raise ValueError(engine)


def build_command(engine, model, port, cold_fields, extra_flags=(), mock_engine="sglang"):
    table = COLD_FLAGS[engine]
    unknown = sorted(k for k in cold_fields if k not in table)
    if unknown:
        raise ValueError(f"no launch flag for cold fields {unknown} on {engine}")
    cmd = base_command(engine, model, port, mock_engine) + list(extra_flags)
    for k, v in sorted(cold_fields.items()):
        cmd += [table[k], str(v)]
    return cmd


class Supervisor:
    def __init__(self, store: Store, manifest: Manifest, *, launcher, adapter_engine, model, port,
                 replica_id, group, extra_flags=(), health_timeout=1800, drain_s=0, env=None, log_path=None):
        self.store, self.manifest = store, manifest
        self.launcher, self.adapter_engine, self.model, self.port = launcher, adapter_engine, model, port
        self.replica_id, self.group, self.extra, self.env = replica_id, group, list(extra_flags), env
        self.health_timeout, self.drain_s, self.log_path = health_timeout, drain_s, log_path
        self.adapter = make_adapter(adapter_engine, f"http://127.0.0.1:{port}")
        self.proc, self.cold_applied, self.applied_version_id = None, None, None
        self.last_reinit_s = None

    def split(self, fields):
        hot, cold = {}, {}
        for k, v in fields.items():
            (cold if self.manifest.knobs.get(k) and self.manifest.knobs[k].klass == COLD else hot)[k] = v
        return hot, cold

    def healthy(self):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    def launch(self, cold_fields):
        cmd = build_command(self.launcher, self.model, self.port, cold_fields, self.extra,
                            mock_engine=self.adapter_engine)
        out = open(self.log_path, "ab") if self.log_path else subprocess.DEVNULL
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=self.env)
        while not self.healthy():
            if self.proc.poll() is not None:
                raise RuntimeError(f"engine exited with {self.proc.returncode} during boot, cmd={cmd}")
            if time.perf_counter() - t0 > self.health_timeout:
                self.stop(); raise RuntimeError("engine did not become healthy in time")
            time.sleep(0.05 if self.launcher == "mock" else 2)
        self.cold_applied = dict(cold_fields)
        return time.perf_counter() - t0

    def stop(self):
        if self.proc and self.proc.poll() is None:
            if self.drain_s:
                time.sleep(self.drain_s)
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill(); self.proc.wait()
        self.proc = None

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def tick(self):
        desired = self.store.desired_for(self.replica_id, self.group)
        if desired is None:
            if not self.alive():
                self.launch(self.cold_applied or {})
            self.store.record_status(self.replica_id, self.group, None, "healthy")
            return "healthy"
        if desired.id == self.applied_version_id and self.alive():
            self.store.record_status(self.replica_id, self.group, desired.id, "healthy")
            return "healthy"

        hot, cold = self.split(desired.fields)
        if not self.alive() or cold != (self.cold_applied or {}):
            self.store.record_status(self.replica_id, self.group, self.applied_version_id, "reinit", f"cold fields {sorted(cold)}")
            self.stop()
            self.last_reinit_s = self.launch(cold)
        if hot:
            accepted, rejected = self.manifest.validate(hot)
            if rejected:
                self.store.record_status(self.replica_id, self.group, self.applied_version_id, "failed", f"manifest rejected {rejected}")
                return "failed"
            r = self.adapter.set_hot(accepted)
            if not r.ok:
                self.store.record_status(self.replica_id, self.group, self.applied_version_id, "failed", f"engine rejected {r.rejected}")
                return "failed"
        self.applied_version_id = desired.id
        detail = f"reinit_s={self.last_reinit_s:.3f}" if self.last_reinit_s is not None else ""
        self.store.record_status(self.replica_id, self.group, desired.id, "healthy", detail)
        return "healthy"

    def run(self, interval_s=1.0, stop=None):
        try:
            while stop is None or not stop.is_set():
                self.tick()
                time.sleep(interval_s)
        finally:
            self.stop()
