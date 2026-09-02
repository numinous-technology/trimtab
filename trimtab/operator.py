"""Kubernetes operator. An InferenceConfig object is the front door to the store.

    apiVersion: trimtab.numinous.technology/v1alpha1
    kind: InferenceConfig
    metadata: {name: prod-cap, namespace: inference}
    spec:
      group: prod
      reason: lower the cap for the evening
      fields: {max_running_requests: 64}
      canary:                      # optional, without it the version is promoted directly
        scope: [r0]
        gates: [{metric: p99_ttft_ms, max_ratio: 1.25}]
        window_s: 300

The operator polls InferenceConfig objects. When spec.fields changes (tracked by
a hash in status), it proposes a version in the store and either promotes it or
opens a canary. It writes the store's view back into status so kubectl shows
what each replica applied. Daemons and supervisors keep talking to the store,
the operator never touches an engine.

Stdlib only. In a pod it reads the service account token and CA. Locally,
point --api at `kubectl proxy`.
"""
import argparse
import hashlib
import json
import os
import ssl
import time
import urllib.error
import urllib.request

from .canary import Canary
from .metrics import SGLANG_MAP, VLLM_MAP, PrometheusMetrics
from .server import open_store

GROUP, VERSION, PLURAL = "trimtab.numinous.technology", "v1alpha1", "inferenceconfigs"


class K8s:
    def __init__(self, api=None, token=None, ca=None):
        self.api = api or f"https://{os.environ['KUBERNETES_SERVICE_HOST']}:{os.environ.get('KUBERNETES_SERVICE_PORT', '443')}"
        self.token = token
        if token is None and os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount/token"):
            self.token = open("/var/run/secrets/kubernetes.io/serviceaccount/token").read().strip()
        self.ctx = None
        ca = ca or ("/var/run/secrets/kubernetes.io/serviceaccount/ca.crt" if os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount/ca.crt") else None)
        if self.api.startswith("https") and ca:
            self.ctx = ssl.create_default_context(cafile=ca)

    def _req(self, method, path, body=None, content_type="application/json"):
        headers = {"Content-Type": content_type}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(self.api + path, method=method, headers=headers,
                                     data=json.dumps(body).encode() if body is not None else None)
        with urllib.request.urlopen(req, timeout=20, context=self.ctx) as r:
            return json.loads(r.read() or b"null")

    def list_configs(self, namespace=None):
        ns = f"/namespaces/{namespace}" if namespace else ""
        return self._req("GET", f"/apis/{GROUP}/{VERSION}{ns}/{PLURAL}").get("items", [])

    def patch_status(self, obj, status):
        ns, name = obj["metadata"]["namespace"], obj["metadata"]["name"]
        return self._req("PATCH", f"/apis/{GROUP}/{VERSION}/namespaces/{ns}/{PLURAL}/{name}/status",
                         {"status": status}, content_type="application/merge-patch+json")


def spec_hash(spec):
    return hashlib.sha256(json.dumps({"fields": spec.get("fields", {}), "canary": spec.get("canary")}, sort_keys=True).encode()).hexdigest()[:16]


class Operator:
    def __init__(self, k8s: K8s, store, engine="sglang", metrics_urls=None, namespace=None, replicas=None):
        self.k8s, self.store, self.engine, self.namespace = k8s, store, engine, namespace
        self.metrics_urls = metrics_urls or {}
        self.replicas = replicas or []

    def reconcile_one(self, obj):
        spec, status = obj.get("spec", {}), obj.get("status", {}) or {}
        group = spec["group"]
        h = spec_hash(spec)
        new_status = dict(status)
        if status.get("appliedHash") != h and spec.get("fields"):
            v = self.store.propose(group, spec["fields"], f"k8s/{obj['metadata']['namespace']}/{obj['metadata']['name']}", spec.get("reason", ""))
            new_status.update(appliedHash=h, versionId=v.id)
            if spec.get("canary"):
                c = spec["canary"]
                replicas = self.replicas or sorted({s["replica_id"] for s in self.store.status(group)})
                canary = Canary(self.store, PrometheusMetrics(self.metrics_urls, SGLANG_MAP if self.engine == "sglang" else VLLM_MAP), group, replicas)
                cid = canary.start(v.id, c["scope"], c["gates"], float(c.get("window_s", 300)))
                new_status.update(phase="canary", canaryId=cid)
            else:
                self.store.promote(group, v.id)
                new_status.update(phase="promoted")
        if new_status.get("phase") == "canary" and new_status.get("canaryId"):
            c = self.store.canary(new_status["canaryId"])
            if c["status"] != "observing":
                new_status.update(phase=c["status"], canaryDecision=c["decision"])
        d = self.store.desired(group)
        new_status.update(desiredVersionId=d.id if d else None,
                          replicas=[{"replica": s["replica_id"], "appliedVersionId": s["applied_version_id"], "state": s["state"], "detail": s["detail"]}
                                    for s in self.store.status(group)])
        if new_status != status:
            self.k8s.patch_status(obj, new_status)
        return new_status

    def tick(self):
        out = {}
        for obj in self.k8s.list_configs(self.namespace):
            try:
                out[obj["metadata"]["name"]] = self.reconcile_one(obj)
            except (ValueError, KeyError, urllib.error.HTTPError) as e:
                self.k8s.patch_status(obj, {"phase": "error", "error": str(e)})
                out[obj["metadata"]["name"]] = {"phase": "error", "error": str(e)}
        return out

    def run(self, interval_s=5.0, stop=None):
        while stop is None or not stop.is_set():
            self.tick()
            time.sleep(interval_s)


def main():
    ap = argparse.ArgumentParser(prog="trimtab.operator")
    ap.add_argument("--db", required=True, help="sqlite, postgresql://, or http:// trimtab.server")
    ap.add_argument("--token", default=os.environ.get("TRIMTAB_TOKEN"))
    ap.add_argument("--api", help="kube api url, defaults to in-cluster")
    ap.add_argument("--namespace")
    ap.add_argument("--engine", default="sglang", choices=["sglang", "vllm"])
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    op = Operator(K8s(a.api), open_store(a.db, a.token), a.engine, namespace=a.namespace)
    if a.once:
        print(json.dumps(op.tick(), indent=1)); return
    op.run(a.interval)


if __name__ == "__main__":
    main()
