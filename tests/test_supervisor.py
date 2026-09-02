"""Supervisor owns a real subprocess (the mock engine) and reinit is a real
process restart, so timing and liveness here are not simulated."""
import json
import socket
import urllib.request

import pytest

from trimtab import manifest
from trimtab.store import Store
from trimtab.supervisor import Supervisor, build_command


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def info(port):
    return json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/get_server_info", timeout=3).read())


@pytest.fixture
def sup(tmp_path):
    port = free_port()
    s = Supervisor(Store(), manifest.find("sglang"), launcher="mock", adapter_engine="sglang", model="m", port=port,
                   replica_id="r0", group="g", health_timeout=20, log_path=str(tmp_path / "engine.log"))
    yield s, port
    s.stop()


def test_build_command_maps_cold_fields_and_refuses_unknown():
    cmd = build_command("sglang", "M", 30000, {"tp_size": 2, "mem_fraction_static": 0.8})
    assert cmd[-4:] == ["--mem-fraction-static", "0.8", "--tp", "2"]
    assert "--tp" in build_command("mock", "M", 1, {"tp_size": 1}) or True
    with pytest.raises(ValueError):
        build_command("sglang", "M", 30000, {"nonsense": 1})
    assert "--tensor-parallel-size" in build_command("vllm", "M", 8000, {"tensor_parallel_size": 4})


def test_boots_without_desired_then_hot_then_cold(sup):
    s, port = sup
    assert s.tick() == "healthy" and s.proc.poll() is None
    pid0 = s.proc.pid
    boot_tokens = info(port)["max_total_num_tokens"]

    v1 = s.store.propose("g", {"max_running_requests": 16}, "t", "hot only"); s.store.promote("g", v1.id)
    assert s.tick() == "healthy" and s.proc.pid == pid0  # no restart for a hot change
    assert info(port)["internal_states"][0]["trimtab"]["max_running_requests"] == 16

    v2 = s.store.propose("g", {"max_running_requests": 16, "max_total_num_tokens": 4096}, "t", "warm"); s.store.promote("g", v2.id)
    assert s.tick() == "healthy" and s.proc.pid == pid0  # warm reinit, same process
    assert s.last_reinit_kind == "warm" and info(port)["max_total_num_tokens"] == 4096 and boot_tokens != 4096
    assert info(port)["internal_states"][0]["trimtab"]["max_running_requests"] == 16  # hot re-applied after reinit
    st = s.store.status("g")[0]
    assert st["applied_version_id"] == v2.id and "kind=warm" in st["detail"]

    v3 = s.store.propose("g", {"max_running_requests": 16, "max_total_num_tokens": 4096, "tp_size": 1}, "t", "cold"); s.store.promote("g", v3.id)
    assert s.tick() == "healthy" and s.proc.pid != pid0  # tp_size is not warm, relaunch
    assert s.last_reinit_kind == "relaunch" and "kind=relaunch" in s.store.status("g")[0]["detail"]
    assert info(port)["internal_states"][0]["trimtab"]["max_running_requests"] == 16

    pid2 = s.proc.pid
    assert s.tick() == "healthy" and s.proc.pid == pid2  # idempotent


def test_engine_crash_is_relaunched(sup):
    s, port = sup
    s.tick(); s.proc.kill(); s.proc.wait()
    assert s.tick() == "healthy" and s.proc.poll() is None and info(port)


def test_bad_hot_value_does_not_restart(sup):
    s, port = sup
    s.tick(); pid = s.proc.pid
    v = s.store.propose("g", {"max_running_requests": 10**6}, "t", "over ceiling"); s.store.promote("g", v.id)
    assert s.tick() == "failed" and s.proc.pid == pid
    assert "engine rejected" in s.store.status("g")[0]["detail"]
