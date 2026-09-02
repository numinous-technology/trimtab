"""Operator against a real Kubernetes API server (kind). Skips when no
kubectl proxy is reachable. The test fixture expects `kubectl proxy --port
8001` running against a cluster with deploy/k8s/crd.yaml applied."""
import json
import subprocess
import time
import urllib.request

import pytest

from trimtab.operator import K8s, Operator
from trimtab.store import Store

API = "http://127.0.0.1:8001"


def _api_up():
    try:
        urllib.request.urlopen(f"{API}/apis/trimtab.numinous.technology/v1alpha1", timeout=2)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _api_up(), reason="no kubectl proxy on 8001 with the trimtab CRD")


def kubectl(*args, input=None):
    r = subprocess.run(["kubectl", *args], capture_output=True, text=True, input=input)
    assert r.returncode == 0, r.stderr
    return r.stdout


@pytest.fixture
def cfg():
    name = f"t-{int(time.time()) % 100000}"
    kubectl("apply", "-f", "-", input=json.dumps({
        "apiVersion": "trimtab.numinous.technology/v1alpha1", "kind": "InferenceConfig",
        "metadata": {"name": name, "namespace": "default"},
        "spec": {"group": "g", "reason": "test", "fields": {"max_running_requests": 16}},
    }))
    yield name
    kubectl("delete", "inferenceconfig", name, "--ignore-not-found")


def test_inferenceconfig_becomes_a_promoted_version(cfg):
    store = Store()
    op = Operator(K8s(API), store)
    out = op.tick()
    assert out[cfg]["phase"] == "promoted"
    d = store.desired("g")
    assert d.fields == {"max_running_requests": 16} and d.created_by == f"k8s/default/{cfg}"
    live = json.loads(kubectl("get", "inferenceconfig", cfg, "-o", "json"))["status"]
    assert live["phase"] == "promoted" and live["desiredVersionId"] == d.id

    op.tick()  # same spec, no new version
    assert len(store.versions("g")) == 1

    kubectl("patch", "inferenceconfig", cfg, "--type", "merge", "-p", json.dumps({"spec": {"fields": {"max_running_requests": 8}}}))
    op.tick()
    assert len(store.versions("g")) == 2 and store.desired("g").fields == {"max_running_requests": 8}

    store.record_status("r0", "g", store.desired("g").id, "healthy", "")
    op.tick()
    live = json.loads(kubectl("get", "inferenceconfig", cfg, "-o", "json"))["status"]
    assert live["replicas"][0]["replica"] == "r0" and live["replicas"][0]["state"] == "healthy"


def test_canary_spec_opens_a_canary(cfg):
    store = Store()
    store.record_status("r0", "g", None, "healthy"); store.record_status("r1", "g", None, "healthy")
    base = store.propose("g", {"max_running_requests": 32}, "t", "base"); store.promote("g", base.id)
    kubectl("patch", "inferenceconfig", cfg, "--type", "merge", "-p", json.dumps(
        {"spec": {"fields": {"max_running_requests": 4}, "canary": {"scope": ["r0"], "gates": [{"metric": "x", "max_abs": 1}], "window_s": 10}}}))
    op = Operator(K8s(API), store)
    out = op.tick()
    assert out[cfg]["phase"] == "canary"
    assert store.desired("g").id == base.id  # group untouched during canary
    assert store.desired_for("r0", "g").fields == {"max_running_requests": 4}
    assert store.desired_for("r1", "g").id == base.id
