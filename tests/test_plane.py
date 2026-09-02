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


NEW_KNOBS = {
    "sglang": [("max_prefill_tokens", 4096), ("schedule_policy", "lpm"), ("schedule_conservativeness", 0.5), ("log_level", "WARNING")],
    "vllm": [("long_prefill_token_threshold", 2048), ("log_level", "DEBUG")],
}
BAD_KNOBS = {
    "sglang": [("schedule_policy", "bogus"), ("schedule_conservativeness", 0), ("log_level", "LOUD")],
    "vllm": [("long_prefill_token_threshold", -5), ("log_level", "LOUD")],
}


def test_new_hot_knobs_apply_and_read_back(engine):
    name, url, state = engine
    a = make_adapter(name, url)
    m = manifest.find(name)
    for k, v in NEW_KNOBS[name]:
        accepted, rejected = m.validate({k: v})
        assert accepted == {k: v}, rejected
        r = a.set_hot({k: v})
        assert r.ok, (k, r.rejected)
        assert str(a.read_knobs()[k]).lower() == str(v).lower()


def test_new_hot_knobs_bad_values_rejected_at_manifest_and_engine(engine):
    name, url, state = engine
    a = make_adapter(name, url)
    m = manifest.find(name)
    before = dict(state.knobs)
    for k, v in BAD_KNOBS[name]:
        accepted, rejected = m.validate({k: v})
        assert not accepted and k in rejected
        assert not a.set_hot({k: v}).ok  # engine agrees even when the manifest is bypassed
    assert state.knobs == before


def test_warm_reinit_on_sglang_adapter(engine):
    name, url, state = engine
    if name != "sglang":
        pytest.skip("warm reinit is an sglang path today")
    a = make_adapter(name, url)
    before = a.read_raw()["max_total_num_tokens"]
    r = a.reinit_warm({"max_total_tokens": before // 2, "max_running_requests": 4})
    assert r.ok and r.applied["last_reinit"]["max_total_num_tokens"] == before // 2
    assert a.read_knobs()["max_running_requests"] == 4
    assert not a.reinit_warm({"tp_size": 2}).ok


def test_warm_reinit_on_vllm_adapter(engine):
    name, url, state = engine
    if name != "vllm":
        pytest.skip("vllm path")
    a = make_adapter(name, url)
    r = a.reinit_warm({"gpu_memory_utilization": 0.5, "max_num_seqs": 4})
    assert r.ok and r.applied["last_reinit"]["ok"] and a.read_knobs()["max_num_seqs"] == 4
    assert not a.reinit_warm({"tensor_parallel_size": 2}).ok
