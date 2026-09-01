"""trimtab command line.

    trimtab set    --engine sglang --url http://host:30000 max_running_requests=64
    trimtab get    --engine sglang --url http://host:30000
    trimtab propose --db trimtab.db --group prod max_running_requests=64 --reason "lower cap"
    trimtab promote --db trimtab.db --group prod 3
    trimtab rollback --db trimtab.db --group prod 2
    trimtab versions --db trimtab.db --group prod
    trimtab status  --db trimtab.db --group prod
    trimtab daemon  --db trimtab.db --group prod --engine sglang --url http://127.0.0.1:30000 --replica r0

`set` and `get` talk to one engine directly with no store, for quick
experiments. The store verbs are the versioned path. `daemon` is trimtabd's
hot path, a reconciler loop against one engine.
"""
import argparse
import getpass
import json
import sys

from . import manifest
from .adapters import make_adapter
from .canary import Canary
from .metrics import SGLANG_MAP, VLLM_MAP, PrometheusMetrics
from .reconciler import Reconciler
from .store import Store
from .supervisor import Supervisor


def parse_kv(items):
    out = {}
    for it in items:
        k, _, v = it.partition("=")
        if not _:
            sys.exit(f"expected key=value, got {it!r}")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def cmd_set(a):
    m = manifest.find(a.engine)
    accepted, rejected = m.validate(parse_kv(a.kv))
    if rejected:
        print(json.dumps({"rejected_by_manifest": rejected}, indent=1))
        if not accepted:
            sys.exit(1)
    r = make_adapter(a.engine, a.url).set_hot(accepted)
    print(json.dumps({"ok": r.ok, "applied": r.applied, "rejected": r.rejected,
                      "latency_ms": round(r.latency_ms, 2), "http_status": r.http_status}, indent=1))
    sys.exit(0 if r.ok else 1)


def cmd_get(a):
    print(json.dumps(make_adapter(a.engine, a.url).read_knobs(), indent=1))


def cmd_propose(a):
    v = Store(a.db).propose(a.group, parse_kv(a.kv), getpass.getuser(), a.reason)
    print(json.dumps({"version_id": v.id, "parent_id": v.parent_id, "fields": v.fields}))


def cmd_promote(a):
    Store(a.db).promote(a.group, a.version)
    print(json.dumps({"group": a.group, "desired_version_id": a.version}))


def cmd_rollback(a):
    Store(a.db).rollback(a.group, a.version)
    print(json.dumps({"group": a.group, "desired_version_id": a.version, "rollback": True}))


def cmd_versions(a):
    s = Store(a.db)
    d = s.desired(a.group)
    for v in s.versions(a.group):
        mark = "*" if d and v.id == d.id else " "
        print(f"{mark} {v.id:>4}  parent={v.parent_id!s:>4}  {v.created_by:<10} {v.reason:<30} {json.dumps(v.fields)}")


def cmd_status(a):
    print(json.dumps(Store(a.db).status(a.group), indent=1))


def _canary(a):
    urls = parse_kv(a.metrics or [])
    metrics = PrometheusMetrics(urls, SGLANG_MAP if a.engine == "sglang" else VLLM_MAP)
    replicas = a.replicas.split(",")
    return Canary(Store(a.db), metrics, a.group, replicas)


def cmd_canary_start(a):
    gates = json.loads(a.gates)
    cid = _canary(a).start(a.version, a.scope.split(","), gates, a.window)
    print(json.dumps({"canary_id": cid, "candidate_version_id": a.version, "scope": a.scope.split(","), "window_s": a.window}))


def cmd_canary_observe(a):
    decision, findings = _canary(a).observe(a.canary_id, samples=a.samples)
    print(json.dumps({"canary_id": a.canary_id, "decision": decision, "findings": findings}, indent=1))
    sys.exit(0 if decision == "promoted" else 1)


def cmd_canary_abort(a):
    _canary(a).abort(a.canary_id)
    print(json.dumps({"canary_id": a.canary_id, "decision": "reverted"}))


def cmd_canary_show(a):
    print(json.dumps(Store(a.db).canary(a.canary_id), indent=1))


def cmd_daemon(a):
    rec = Reconciler(Store(a.db), manifest.find(a.engine), make_adapter(a.engine, a.url), a.replica, a.group)
    if a.once:
        print(rec.tick())
        return
    rec.run(a.interval)


def cmd_supervise(a):
    sup = Supervisor(Store(a.db), manifest.find(a.engine), launcher=a.launcher or a.engine, adapter_engine=a.engine,
                     model=a.model, port=a.port, replica_id=a.replica, group=a.group,
                     extra_flags=a.extra_flag or [], drain_s=a.drain, log_path=a.engine_log)
    if a.once:
        print(sup.tick()); sup.stop(); return
    sup.run(a.interval)


def main(argv=None):
    p = argparse.ArgumentParser(prog="trimtab")
    sub = p.add_subparsers(dest="cmd", required=True)

    def eng(sp):
        sp.add_argument("--engine", required=True, choices=sorted(manifest_engines()))
        sp.add_argument("--url", required=True)

    def db(sp):
        sp.add_argument("--db", required=True)
        sp.add_argument("--group", required=True)

    s = sub.add_parser("set"); eng(s); s.add_argument("kv", nargs="+"); s.set_defaults(f=cmd_set)
    s = sub.add_parser("get"); eng(s); s.set_defaults(f=cmd_get)
    s = sub.add_parser("propose"); db(s); s.add_argument("kv", nargs="+"); s.add_argument("--reason", required=True); s.set_defaults(f=cmd_propose)
    s = sub.add_parser("promote"); db(s); s.add_argument("version", type=int); s.set_defaults(f=cmd_promote)
    s = sub.add_parser("rollback"); db(s); s.add_argument("version", type=int); s.set_defaults(f=cmd_rollback)
    s = sub.add_parser("versions"); db(s); s.set_defaults(f=cmd_versions)
    s = sub.add_parser("status"); db(s); s.set_defaults(f=cmd_status)
    s = sub.add_parser("daemon"); db(s); eng(s); s.add_argument("--replica", required=True)
    s.add_argument("--interval", type=float, default=1.0); s.add_argument("--once", action="store_true"); s.set_defaults(f=cmd_daemon)

    s = sub.add_parser("supervise", help="trimtabd, own the engine process and converge it on desired state")
    db(s); s.add_argument("--engine", required=True, choices=sorted(manifest_engines()))
    s.add_argument("--launcher", choices=["sglang", "vllm", "mock"], help="defaults to --engine, mock for tests")
    s.add_argument("--model", required=True); s.add_argument("--port", type=int, required=True)
    s.add_argument("--replica", required=True); s.add_argument("--extra-flag", action="append", help="passed to the engine verbatim")
    s.add_argument("--drain", type=float, default=0); s.add_argument("--engine-log", default="engine.log")
    s.add_argument("--interval", type=float, default=1.0); s.add_argument("--once", action="store_true"); s.set_defaults(f=cmd_supervise)

    def can(sp):
        db(sp)
        sp.add_argument("--engine", required=True, choices=sorted(manifest_engines()))
        sp.add_argument("--replicas", required=True, help="comma separated replica ids in the group")
        sp.add_argument("--metrics", nargs="*", help="replica=http://host:port/metrics")

    c = sub.add_parser("canary"); cs = c.add_subparsers(dest="sub", required=True)
    s = cs.add_parser("start"); can(s); s.add_argument("version", type=int); s.add_argument("--scope", required=True)
    s.add_argument("--gates", required=True, help="json list of gates"); s.add_argument("--window", type=float, default=300); s.set_defaults(f=cmd_canary_start)
    s = cs.add_parser("observe"); can(s); s.add_argument("canary_id", type=int); s.add_argument("--samples", type=int, default=5); s.set_defaults(f=cmd_canary_observe)
    s = cs.add_parser("abort"); can(s); s.add_argument("canary_id", type=int); s.set_defaults(f=cmd_canary_abort)
    s = cs.add_parser("show"); db(s); s.add_argument("canary_id", type=int); s.set_defaults(f=cmd_canary_show)

    a = p.parse_args(argv)
    a.f(a)


def manifest_engines():
    from pathlib import Path
    return [d.name for d in (Path(__file__).resolve().parent.parent / "engine").iterdir() if (d / "manifests").is_dir()]


if __name__ == "__main__":
    main()
