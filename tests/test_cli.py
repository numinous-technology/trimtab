"""Drive the CLI as a user would, against a mock engine and a real SQLite file."""
import json
import subprocess
import sys
import threading

import pytest

from trimtab.mock_engine import serve


@pytest.fixture
def mock():
    srv = serve("sglang", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", srv.state
    srv.shutdown()


def tt(*args):
    r = subprocess.run([sys.executable, "-m", "trimtab.cli", *args], capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def test_set_get_direct(mock):
    url, state = mock
    code, out, _ = tt("set", "--engine", "sglang", "--url", url, "max_running_requests=32")
    assert code == 0 and json.loads(out)["applied"] == {"max_running_requests": 32}
    assert state.knobs["max_running_requests"] == 32
    code, out, _ = tt("get", "--engine", "sglang", "--url", url)
    assert json.loads(out)["max_running_requests"] == 32


def test_set_cold_knob_exits_nonzero(mock):
    url, state = mock
    code, out, _ = tt("set", "--engine", "sglang", "--url", url, "tp_size=4")
    assert code == 1 and "cold" in json.loads(out)["rejected_by_manifest"]["tp_size"]


def test_versioned_flow_with_daemon(mock, tmp_path):
    url, state = mock
    db = str(tmp_path / "t.db")
    boot = state.knobs["max_running_requests"]
    _, out, _ = tt("propose", "--db", db, "--group", "g", "--reason", "baseline", f"max_running_requests={boot}")
    v1 = json.loads(out)["version_id"]
    _, out, _ = tt("propose", "--db", db, "--group", "g", "--reason", "experiment", "max_running_requests=8")
    v2 = json.loads(out)["version_id"]
    assert tt("promote", "--db", db, "--group", "g", str(v2))[0] == 0
    code, out, err = tt("daemon", "--db", db, "--group", "g", "--engine", "sglang", "--url", url, "--replica", "r0", "--once")
    assert code == 0 and out.strip() == "healthy", err
    assert state.knobs["max_running_requests"] == 8
    tt("rollback", "--db", db, "--group", "g", str(v1))
    tt("daemon", "--db", db, "--group", "g", "--engine", "sglang", "--url", url, "--replica", "r0", "--once")
    assert state.knobs["max_running_requests"] == boot
    _, out, _ = tt("versions", "--db", db, "--group", "g")
    assert out.count("\n") == 2 and out.startswith("*")
    _, out, _ = tt("status", "--db", db, "--group", "g")
    assert json.loads(out)[0]["applied_version_id"] == v1


def test_canary_start_and_abort_via_cli(mock, tmp_path):
    url, state = mock
    db = str(tmp_path / "c.db")
    boot = state.knobs["max_running_requests"]
    _, out, _ = tt("propose", "--db", db, "--group", "g", "--reason", "b", f"max_running_requests={boot}")
    v1 = json.loads(out)["version_id"]; tt("promote", "--db", db, "--group", "g", str(v1))
    _, out, _ = tt("propose", "--db", db, "--group", "g", "--reason", "c", "max_running_requests=8")
    v2 = json.loads(out)["version_id"]
    common = ["--db", db, "--group", "g", "--engine", "sglang", "--replicas", "r0,r1"]
    code, out, err = tt("canary", "start", *common, str(v2), "--scope", "r0", "--gates", '[{"metric":"error_rate","max_abs":0.01}]')
    assert code == 0, err
    cid = json.loads(out)["canary_id"]
    tt("daemon", "--db", db, "--group", "g", "--engine", "sglang", "--url", url, "--replica", "r0", "--once")
    assert state.knobs["max_running_requests"] == 8  # r0 is the canary replica
    _, out, _ = tt("canary", "show", "--db", db, "--group", "g", str(cid))
    assert json.loads(out)["status"] == "observing"
    assert tt("canary", "abort", *common, str(cid))[0] == 0
    tt("daemon", "--db", db, "--group", "g", "--engine", "sglang", "--url", url, "--replica", "r0", "--once")
    assert state.knobs["max_running_requests"] == boot
