"""Network API for the store, and a RemoteStore client with the Store interface.

The daemon, supervisor, and canary take a Store. Give them a RemoteStore and
they talk to this server instead of a local file, unchanged. One bearer token
guards every route. JSON in, JSON out, stdlib only.

    python3 -m trimtab.server --db postgresql://... --port 7070 --token secret
    python3 -m trimtab.cli daemon --db http://plane:7070 --token secret ...

Routes
    POST /v1/propose        {group, fields, created_by, reason}      -> version
    POST /v1/promote        {group, version_id}
    POST /v1/rollback       {group, version_id}
    GET  /v1/desired        ?group=&replica=                         -> version or null
    GET  /v1/version        ?id=
    GET  /v1/versions       ?group=
    POST /v1/status         {replica_id, group, applied_version_id, state, detail}
    GET  /v1/status         ?group=
    POST /v1/canary/open    {group, candidate_id, scope, gates, window_s} -> {id}
    GET  /v1/canary         ?id=
    POST /v1/canary/close   {id, status, decision}
    POST /v1/overrides      {replica_ids, version_id, canary_id}
    DELETE /v1/overrides    ?canary_id=
    GET  /health
"""
import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .store import ConfigVersion, Store


def _v(v):
    return asdict(v) if v is not None else None


def make_handler(store: Store, token: str):
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

        def _auth(self):
            if self.path == "/health":
                return True
            if self.headers.get("Authorization") == f"Bearer {token}":
                return True
            self._json(401, {"error": "bad token"})
            return False

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def _route(self, method):
            if not self._auth():
                return
            u = urllib.parse.urlparse(self.path)
            q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
            b = self._body() if method in ("POST", "DELETE") else {}
            try:
                r = self.dispatch(method, u.path, q, b)
            except KeyError as e:
                return self._json(404, {"error": str(e)})
            except ValueError as e:
                return self._json(400, {"error": str(e)})
            self._json(200, r)

        def dispatch(self, method, path, q, b):
            if path == "/health":
                return {"ok": True}
            if (method, path) == ("POST", "/v1/propose"):
                return _v(store.propose(b["group"], b["fields"], b.get("created_by", "api"), b.get("reason", "")))
            if (method, path) == ("POST", "/v1/promote"):
                store.promote(b["group"], int(b["version_id"])); return {"ok": True}
            if (method, path) == ("POST", "/v1/rollback"):
                store.rollback(b["group"], int(b["version_id"])); return {"ok": True}
            if (method, path) == ("GET", "/v1/desired"):
                v = store.desired_for(q["replica"], q["group"]) if q.get("replica") else store.desired(q["group"])
                return _v(v)
            if (method, path) == ("GET", "/v1/version"):
                return _v(store.version(int(q["id"])))
            if (method, path) == ("GET", "/v1/versions"):
                return [_v(v) for v in store.versions(q["group"])]
            if (method, path) == ("POST", "/v1/status"):
                store.record_status(b["replica_id"], b["group"], b.get("applied_version_id"), b["state"], b.get("detail", "")); return {"ok": True}
            if (method, path) == ("GET", "/v1/status"):
                return store.status(q["group"])
            if (method, path) == ("POST", "/v1/canary/open"):
                return {"id": store.open_canary(b["group"], int(b["candidate_id"]), b["scope"], b["gates"], float(b["window_s"]))}
            if (method, path) == ("GET", "/v1/canary"):
                return store.canary(int(q["id"]))
            if (method, path) == ("POST", "/v1/canary/close"):
                store.close_canary(int(b["id"]), b["status"], b.get("decision", "")); return {"ok": True}
            if (method, path) == ("POST", "/v1/overrides"):
                store.set_overrides(b["replica_ids"], int(b["version_id"]), int(b["canary_id"])); return {"ok": True}
            if (method, path) == ("DELETE", "/v1/overrides"):
                store.clear_overrides(int(q["canary_id"])); return {"ok": True}
            raise KeyError(f"no route {method} {path}")

        def do_GET(self): self._route("GET")
        def do_POST(self): self._route("POST")
        def do_DELETE(self): self._route("DELETE")

    return H


def serve(store, port, token, host="0.0.0.0"):
    ThreadingHTTPServer.daemon_threads = True
    ThreadingHTTPServer.request_queue_size = 256
    return ThreadingHTTPServer((host, port), make_handler(store, token))


class RemoteStore:
    """Same interface as Store, over HTTP. Errors from the server become the
    same exception types the local Store raises."""

    def __init__(self, base_url, token, timeout=15):
        self.base, self.token, self.timeout = base_url.rstrip("/"), token, timeout

    def _call(self, method, path, body=None, **q):
        url = self.base + path + (("?" + urllib.parse.urlencode(q)) if q else "")
        req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            msg = json.loads(e.read() or b"{}").get("error", str(e))
            raise {400: ValueError, 404: KeyError, 401: PermissionError}.get(e.code, RuntimeError)(msg)

    @staticmethod
    def _cv(d):
        return ConfigVersion(**d) if d else None

    def propose(self, group, fields, created_by, reason):
        return self._cv(self._call("POST", "/v1/propose", dict(group=group, fields=fields, created_by=created_by, reason=reason)))

    def promote(self, group, version_id): self._call("POST", "/v1/promote", dict(group=group, version_id=version_id))
    def rollback(self, group, version_id): self._call("POST", "/v1/rollback", dict(group=group, version_id=version_id))
    def desired(self, group): return self._cv(self._call("GET", "/v1/desired", group=group))
    def desired_for(self, replica_id, group): return self._cv(self._call("GET", "/v1/desired", group=group, replica=replica_id))
    def version(self, version_id): return self._cv(self._call("GET", "/v1/version", id=version_id))
    def versions(self, group): return [self._cv(d) for d in self._call("GET", "/v1/versions", group=group)]

    def record_status(self, replica_id, group, applied_version_id, state, detail=""):
        self._call("POST", "/v1/status", dict(replica_id=replica_id, group=group, applied_version_id=applied_version_id, state=state, detail=detail))

    def status(self, group): return self._call("GET", "/v1/status", group=group)

    def open_canary(self, group, candidate_id, scope, gates, window_s):
        return self._call("POST", "/v1/canary/open", dict(group=group, candidate_id=candidate_id, scope=list(scope), gates=gates, window_s=window_s))["id"]

    def canary(self, canary_id): return self._call("GET", "/v1/canary", id=canary_id)
    def close_canary(self, canary_id, status, decision): self._call("POST", "/v1/canary/close", dict(id=canary_id, status=status, decision=decision))
    def set_overrides(self, replica_ids, version_id, canary_id): self._call("POST", "/v1/overrides", dict(replica_ids=list(replica_ids), version_id=version_id, canary_id=canary_id))
    def clear_overrides(self, canary_id): self._call("DELETE", "/v1/overrides", canary_id=canary_id)


def open_store(db, token=None):
    """Store(path or dsn) or RemoteStore(http url). The CLI's --db takes any of the three."""
    if str(db).startswith(("http://", "https://")):
        if not token:
            raise ValueError("a remote store needs --token")
        return RemoteStore(db, token)
    return Store(db)


def main():
    ap = argparse.ArgumentParser(prog="trimtab.server")
    ap.add_argument("--db", required=True, help="sqlite path or postgresql:// dsn")
    ap.add_argument("--port", type=int, default=7070)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--token", required=True)
    a = ap.parse_args()
    srv = serve(Store(a.db), a.port, a.token, a.host)
    print(f"trimtab control plane on {a.host}:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
