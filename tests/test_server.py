"""The plane over the network. A real trimtab.server in a thread, a RemoteStore
client, and the reconciler, canary, and CLI driving a mock engine through it."""
import json
import subprocess
import sys
import threading

import pytest

from trimtab import manifest
from trimtab.adapters import make_adapter
from trimtab.canary import PASS, Canary
from trimtab.mock_engine import serve as serve_engine
from trimtab.reconciler import HEALTHY, Reconciler
from trimtab.server import RemoteStore, serve
from trimtab.store import Store

TOKEN = "t0k3n"


@pytest.fixture
def plane():
    srv = serve(Store(), 0, TOKEN, host="127.0.0.1")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.fixture
def engine():
    e = serve_engine("sglang", 0)
    threading.Thread(target=e.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{e.server_address[1]}", e.state
    e.shutdown()


def test_remote_store_contract(plane):
    s = RemoteStore(plane, TOKEN)
    v1 = s.propose("g", {"max_running_requests": 16}, "t", "one")
    assert v1.id and v1.fields == {"max_running_requests": 16} and s.desired("g") is None
    s.promote("g", v1.id)
    assert s.desired("g").id == v1.id and s.desired_for("r0", "g").id == v1.id
    v2 = s.propose("g", {"max_running_requests": 8}, "t", "two")
    assert v2.parent_id == v1.id and [v.id for v in s.versions("g")] == [v1.id, v2.id]
    cid = s.open_canary("g", v2.id, ["r0"], [{"metric": "x", "max_abs": 1}], 5)
    s.set_overrides(["r0"], v2.id, cid)
    assert s.desired_for("r0", "g").id == v2.id and s.desired_for("r1", "g").id == v1.id
    s.clear_overrides(cid); s.close_canary(cid, "reverted", "x")
    assert s.canary(cid)["status"] == "reverted"
    s.record_status("r0", "g", v1.id, "healthy", "")
    assert s.status("g")[0]["applied_version_id"] == v1.id
    with pytest.raises(ValueError):
        s.promote("other", v1.id)
    with pytest.raises(KeyError):
        s.version(99999)


def test_bad_token_rejected(plane):
    with pytest.raises(PermissionError):
        RemoteStore(plane, "wrong").desired("g")


def test_reconciler_and_canary_over_the_network(plane, engine):
    url, state = engine
    s = RemoteStore(plane, TOKEN)
    m = manifest.find("sglang")
    recs = {r: Reconciler(s, m, make_adapter("sglang", url), r, "g") for r in ("r0", "r1")}
    boot = state.knobs["max_running_requests"]
    base = s.propose("g", {"max_running_requests": boot}, "t", "base"); s.promote("g", base.id)
    for r in recs.values(): assert r.tick() == HEALTHY
    cand = s.propose("g", {"max_running_requests": 8}, "t", "cand")

    class M:
        def sample(self, rid): return {"x": 0.0}
    c = Canary(s, M(), "g", recs.keys())
    cid = c.start(cand.id, ["r0"], [{"metric": "x", "max_abs": 1}], 1)
    recs["r0"].tick(); assert state.knobs["max_running_requests"] == 8
    decision, _ = c.observe(cid, samples=1, sleep=lambda x: None)
    assert decision == PASS and s.desired("g").id == cand.id


def test_cli_against_http_store(plane, engine):
    url, state = engine
    def tt(*args):
        r = subprocess.run([sys.executable, "-m", "trimtab.cli", *args], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return r.stdout
    common = ["--db", plane, "--token", TOKEN, "--group", "g"]
    v = json.loads(tt("propose", *common, "--reason", "r", "max_running_requests=12"))["version_id"]
    tt("promote", *common, str(v))
    assert tt("daemon", *common, "--engine", "sglang", "--url", url, "--replica", "r0", "--once").strip() == "healthy"
    assert state.knobs["max_running_requests"] == 12
    assert json.loads(tt("status", *common))[0]["applied_version_id"] == v
