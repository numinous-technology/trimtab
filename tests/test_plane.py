"""End to end on CPU. Mock engine, adapter, manifest, store, reconciler."""
import threading

import pytest

from trimtab import manifest
from trimtab.adapters import make_adapter
from trimtab.mock_engine import serve
from trimtab.reconciler import FAILED, HEALTHY, NEEDS_REINIT, Reconciler
from trimtab.store import Store


@pytest.fixture(params=["sglang", "vllm"])
def engine(request):
    srv = serve(request.param, 0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield request.param, f"http://127.0.0.1:{srv.server_address[1]}", srv.state
    srv.shutdown()


CAP = {"sglang": "max_running_requests", "vllm": "max_num_seqs"}


def test_adapter_applies_and_reads_back(engine):
    name, url, state = engine
    a = make_adapter(name, url)
    assert a.healthy()
    boot = a.read_knobs()[CAP[name]]
    r = a.set_hot({CAP[name]: 8})
    assert r.ok and r.applied == {CAP[name]: 8} and r.latency_ms < 500
    assert a.read_knobs()[CAP[name]] == 8
    assert state.knobs[CAP[name]] == 8
    assert a.set_hot({CAP[name]: boot}).ok


def test_adapter_rejects_above_ceiling(engine):
    name, url, state = engine
    a = make_adapter(name, url)
    before = a.read_knobs()[CAP[name]]
    r = a.set_hot({CAP[name]: before * 10})
    assert not r.ok and CAP[name] in r.rejected
    assert a.read_knobs()[CAP[name]] == before


def test_adapter_rejects_unknown_knob(engine):
    name, url, _ = engine
    r = make_adapter(name, url).set_hot({"nonsense": 1})
    assert not r.ok


def test_reconciler_converges_and_is_idempotent(engine):
    name, url, state = engine
    store, m, a = Store(), manifest.find(name), make_adapter(name, url)
    rec = Reconciler(store, m, a, "r0", "g")
    assert rec.tick() == HEALTHY  # nothing desired yet

    v1 = store.propose("g", {CAP[name]: 16}, "test", "lower cap")
    store.promote("g", v1.id)
    assert rec.tick() == HEALTHY and state.knobs[CAP[name]] == 16
    calls = state.applied_calls
    assert rec.tick() == HEALTHY and state.applied_calls == calls  # no re-apply
    assert store.status("g")[0]["applied_version_id"] == v1.id


def test_reconciler_rollback(engine):
    name, url, state = engine
    store, m, a = Store(), manifest.find(name), make_adapter(name, url)
    rec = Reconciler(store, m, a, "r0", "g")
    boot = state.knobs[CAP[name]]
    v1 = store.propose("g", {CAP[name]: boot}, "t", "baseline"); store.promote("g", v1.id); rec.tick()
    v2 = store.propose("g", {CAP[name]: 4}, "t", "experiment"); store.promote("g", v2.id); rec.tick()
    assert state.knobs[CAP[name]] == 4 and v2.parent_id == v1.id
    store.rollback("g", v1.id); assert rec.tick() == HEALTHY
    assert state.knobs[CAP[name]] == boot
    assert [v.id for v in store.versions("g")] == [v1.id, v2.id]  # nothing erased


def test_reconciler_cold_field_reports_needs_reinit(engine):
    name, url, state = engine
    store, m, a = Store(), manifest.find(name), make_adapter(name, url)
    rec = Reconciler(store, m, a, "r0", "g")
    cold = "tp_size" if name == "sglang" else "tensor_parallel_size"
    v = store.propose("g", {CAP[name]: 8, cold: 4}, "t", "mixed"); store.promote("g", v.id)
    assert rec.tick() == NEEDS_REINIT
    assert state.knobs[CAP[name]] == 8  # hot part still applied
    assert "cold fields pending" in store.status("g")[0]["detail"]


def test_reconciler_manifest_rejection_does_not_touch_engine(engine):
    name, url, state = engine
    store, m, a = Store(), manifest.find(name), make_adapter(name, url)
    rec = Reconciler(store, m, a, "r0", "g")
    before, calls = dict(state.knobs), state.applied_calls
    v = store.propose("g", {CAP[name]: "eight"}, "t", "typo"); store.promote("g", v.id)
    assert rec.tick() == FAILED
    assert state.knobs == before and state.applied_calls == calls


def test_store_promote_wrong_group_refused():
    s = Store()
    v = s.propose("a", {"x": 1}, "t", "r")
    with pytest.raises(ValueError):
        s.promote("b", v.id)
