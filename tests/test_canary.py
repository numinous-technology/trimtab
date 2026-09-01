"""Canary over three mock replicas of one engine. One canary, two baseline."""
import threading

import pytest

from trimtab import manifest
from trimtab.adapters import make_adapter
from trimtab.canary import FAIL, PASS, Canary, evaluate
from trimtab.mock_engine import serve
from trimtab.reconciler import Reconciler
from trimtab.store import Store

CAP = "max_running_requests"
GATES = [{"metric": "error_rate", "max_abs": 0.01}, {"metric": "p99_ttft_ms", "max_ratio": 1.25}]


class StubMetrics:
    def __init__(self, per_replica):
        self.per_replica = per_replica

    def sample(self, replica_id):
        return self.per_replica[replica_id]


@pytest.fixture
def fleet():
    srvs = [serve("sglang", 0) for _ in range(3)]
    for s in srvs:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    store, m = Store(), manifest.find("sglang")
    recs = {f"r{i}": Reconciler(store, m, make_adapter("sglang", f"http://127.0.0.1:{s.server_address[1]}"), f"r{i}", "g")
            for i, s in enumerate(srvs)}
    states = {f"r{i}": s.state for i, s in enumerate(srvs)}
    boot = states["r0"].knobs[CAP]
    v_base = store.propose("g", {CAP: boot}, "t", "baseline"); store.promote("g", v_base.id)
    for r in recs.values(): r.tick()
    yield store, recs, states, boot, v_base
    for s in srvs: s.shutdown()


def tick_all(recs):
    for r in recs.values(): r.tick()


def test_canary_pass_promotes_to_group(fleet):
    store, recs, states, boot, v_base = fleet
    cand = store.propose("g", {CAP: 16}, "t", "candidate")
    metrics = StubMetrics({"r0": {"error_rate": 0.0, "p99_ttft_ms": 100}, "r1": {"error_rate": 0.0, "p99_ttft_ms": 100}, "r2": {"error_rate": 0.0, "p99_ttft_ms": 105}})
    c = Canary(store, metrics, "g", recs.keys())
    cid = c.start(cand.id, ["r0"], GATES, window_s=1)
    tick_all(recs)
    assert states["r0"].knobs[CAP] == 16 and states["r1"].knobs[CAP] == boot and states["r2"].knobs[CAP] == boot
    decision, findings = c.observe(cid, samples=2, sleep=lambda s: None)
    assert decision == PASS, findings
    tick_all(recs)
    assert all(s.knobs[CAP] == 16 for s in states.values())
    assert store.desired("g").id == cand.id and store.canary(cid)["status"] == PASS


def test_canary_fail_reverts_canary_replica(fleet):
    store, recs, states, boot, v_base = fleet
    cand = store.propose("g", {CAP: 4}, "t", "too low")
    metrics = StubMetrics({"r0": {"error_rate": 0.0, "p99_ttft_ms": 400}, "r1": {"error_rate": 0.0, "p99_ttft_ms": 100}, "r2": {"error_rate": 0.0, "p99_ttft_ms": 100}})
    c = Canary(store, metrics, "g", recs.keys())
    cid = c.start(cand.id, ["r0"], GATES, window_s=1)
    tick_all(recs)
    assert states["r0"].knobs[CAP] == 4
    decision, findings = c.observe(cid, samples=2, sleep=lambda s: None)
    assert decision == FAIL and any("p99_ttft_ms: ratio" in f for f in findings)
    tick_all(recs)
    assert states["r0"].knobs[CAP] == boot  # back on baseline
    assert store.desired("g").id == v_base.id  # group never moved
    assert [v.id for v in store.versions("g")] == [v_base.id, cand.id]  # candidate kept in history


def test_canary_abort(fleet):
    store, recs, states, boot, _ = fleet
    cand = store.propose("g", {CAP: 8}, "t", "c")
    c = Canary(store, StubMetrics({}), "g", recs.keys())
    cid = c.start(cand.id, ["r1"], GATES, 60)
    tick_all(recs); assert states["r1"].knobs[CAP] == 8
    c.abort(cid); tick_all(recs)
    assert states["r1"].knobs[CAP] == boot and store.canary(cid)["status"] == FAIL


def test_scope_must_leave_baseline(fleet):
    store, recs, *_ = fleet
    c = Canary(store, StubMetrics({}), "g", recs.keys())
    with pytest.raises(ValueError):
        c.start(1, ["r0", "r1", "r2"], GATES, 1)
    with pytest.raises(ValueError):
        c.start(1, ["r9"], GATES, 1)


def test_evaluate_every_gate_reports():
    ok, f = evaluate(GATES, {"error_rate": 0.0, "p99_ttft_ms": 110}, {"p99_ttft_ms": 100})
    assert ok and len(f) == 2
    ok, f = evaluate(GATES, {"error_rate": 0.05, "p99_ttft_ms": 110}, {"p99_ttft_ms": 100})
    assert not ok and "error_rate" in f[0]
    ok, f = evaluate([{"metric": "x", "max_ratio": 2}], {"x": 1}, {})
    assert not ok and "no usable baseline" in f[0]
    ok, f = evaluate([{"metric": "x", "max_abs": 2}], {}, {})
    assert not ok and "no canary sample" in f[0]
