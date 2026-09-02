"""A fake inference engine that speaks the trimtab control surface of SGLang
or vLLM without a GPU.

It models exactly what the real patches model. Three scheduler attributes with
ceilings recorded at boot, validated on write, readable back. Everything in
the control plane is developed and tested against this on CPU. It also serves
a trivial /generate so load-generating tests have something to hit.

    python3 -m trimtab.mock_engine --engine sglang --port 30000
    python3 -m trimtab.mock_engine --engine vllm --port 8000
"""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class State:
    def __init__(self, engine, mrr=128, mqr=1024, cps=8192, tokens=16384):
        self.engine = engine
        self.tokens = tokens
        self.lock = threading.Lock()
        self.knobs = (
            {"max_running_requests": mrr, "max_queued_requests": mqr, "chunked_prefill_size": cps}
            if engine == "sglang"
            else {"max_num_seqs": mrr, "max_num_batched_tokens": tokens}
        )
        self.ceilings = {k: v for k, v in self.knobs.items() if k != "max_queued_requests"}
        self.applied_calls = 0

    def apply(self, changes):
        applied, rejected = {}, {}
        with self.lock:
            for k, v in changes.items():
                if k not in self.knobs:
                    rejected[k] = "unknown knob"
                    continue
                lo = 0 if k == "max_queued_requests" else 1
                hi = self.ceilings.get(k)
                if not isinstance(v, (int, float)) or int(v) < lo or (hi is not None and int(v) > hi):
                    rejected[k] = f"valid range is {lo}..{hi}"
                    continue
                self.knobs[k] = int(v)
                applied[k] = int(v)
            self.applied_calls += 1
        return applied, rejected


def make_handler(state):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            if self.path.startswith("/health"):
                return self._json(200, {"ok": True})
            if state.engine == "sglang" and self.path == "/get_server_info":
                return self._json(200, {
                    "max_total_num_tokens": state.tokens,
                    "internal_states": [{
                        "effective_max_running_requests_per_dp": state.knobs["max_running_requests"],
                        "trimtab": dict(state.knobs, ceilings=state.ceilings),
                    }],
                })
            if state.engine == "vllm" and self.path == "/trimtab/knobs":
                return self._json(200, dict(state.knobs, max_num_seqs_effective=state.knobs["max_num_seqs"],
                                            running=0, ceilings=state.ceilings))
            self._json(404, {"error": "no route"})

        def do_POST(self):
            body = self._body()
            if state.engine == "sglang" and self.path == "/set_internal_state":
                applied, rejected = state.apply(body.get("server_args", {}))
                return self._json(200, [not rejected])  # real SGLang returns one bool per rank
            if state.engine == "vllm" and self.path == "/trimtab/set_knobs":
                applied, rejected = state.apply(body)
                ok = not rejected
                return self._json(200 if ok else 400, {"ok": ok, "applied": applied, "rejected": rejected})
            if self.path in ("/generate", "/v1/completions"):
                return self._json(200, {"text": "ok", "choices": [{"text": "ok"}]})
            self._json(404, {"error": "no route"})

    return H


def serve(engine, port, host="127.0.0.1", tokens=16384):
    state = State(engine, tokens=tokens)
    ThreadingHTTPServer.request_queue_size = 256
    ThreadingHTTPServer.daemon_threads = True
    srv = ThreadingHTTPServer((host, port), make_handler(state))
    srv.state = state
    return srv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["sglang", "vllm"], required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--tokens", type=int, default=16384, help="stands in for a cold boot-time allocation")
    a = ap.parse_args()
    srv = serve(a.engine, a.port, tokens=a.tokens)
    print(f"mock {a.engine} engine on {a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
