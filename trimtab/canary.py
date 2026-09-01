"""Canary orchestrator. Live config without canary is production roulette.

A canary puts a candidate version on a subset of replicas through
replica_override, leaves the rest on the group's desired version as baseline,
observes for a window, and decides against preregistered gates. Pass promotes
the candidate to the whole group and clears the overrides. Fail clears the
overrides, so the canary replicas reconcile back to baseline on their next
tick, and records why.

Gates are data. Each gate names a metric and a bound relative to baseline or
absolute, so policy changes never need code changes.

    {"metric": "error_rate",  "max_abs": 0.01}
    {"metric": "p99_ttft_ms", "max_ratio": 1.25}
    {"metric": "throughput",  "min_ratio": 0.9}

A MetricsSource is anything with sample(replica_id) -> dict of floats. The
orchestrator averages samples per side over the window. The bench harness, a
Prometheus scraper, or a test stub all fit.
"""
import time

from .store import Store

PASS, FAIL = "promoted", "reverted"


def evaluate(gates, canary_metrics, baseline_metrics):
    """Return (passed, findings). Every gate produces a finding, so a reviewer
    sees what was checked, not only what failed."""
    findings, passed = [], True
    for g in gates:
        m = g["metric"]
        c, b = canary_metrics.get(m), baseline_metrics.get(m)
        if c is None:
            findings.append(f"{m}: no canary sample, gate cannot pass"); passed = False; continue
        if "max_abs" in g and c > g["max_abs"]:
            findings.append(f"{m}: canary {c:.4g} above max_abs {g['max_abs']}"); passed = False; continue
        if "min_abs" in g and c < g["min_abs"]:
            findings.append(f"{m}: canary {c:.4g} below min_abs {g['min_abs']}"); passed = False; continue
        if ("max_ratio" in g or "min_ratio" in g):
            if b is None or b == 0:
                findings.append(f"{m}: no usable baseline for a ratio gate"); passed = False; continue
            r = c / b
            if "max_ratio" in g and r > g["max_ratio"]:
                findings.append(f"{m}: ratio {r:.3f} above max_ratio {g['max_ratio']}"); passed = False; continue
            if "min_ratio" in g and r < g["min_ratio"]:
                findings.append(f"{m}: ratio {r:.3f} below min_ratio {g['min_ratio']}"); passed = False; continue
        findings.append(f"{m}: ok (canary {c:.4g}, baseline {b if b is None else format(b, '.4g')})")
    return passed, findings


def _mean(dicts):
    out, n = {}, {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0.0) + float(v); n[k] = n.get(k, 0) + 1
    return {k: out[k] / n[k] for k in out}


class Canary:
    def __init__(self, store: Store, metrics, group, all_replicas):
        self.store, self.metrics, self.group, self.all = store, metrics, group, list(all_replicas)

    def start(self, candidate_version_id, scope, gates, window_s) -> int:
        if not set(scope) <= set(self.all):
            raise ValueError(f"scope {sorted(scope)} not within replicas {sorted(self.all)}")
        if set(scope) == set(self.all):
            raise ValueError("canary scope must leave at least one baseline replica")
        cid = self.store.open_canary(self.group, candidate_version_id, scope, gates, window_s)
        self.store.set_overrides(scope, candidate_version_id, cid)
        return cid

    def observe(self, canary_id, samples=5, sleep=None):
        """Collect samples across the window, then decide. Returns the decision
        string and findings. sleep is injectable for tests."""
        c = self.store.canary(canary_id)
        if c["status"] != "observing":
            raise ValueError(f"canary {canary_id} already {c['status']}")
        scope = set(c["scope"]); base = [r for r in self.all if r not in scope]
        gap = c["window_s"] / max(samples, 1)
        cs, bs = [], []
        for _ in range(samples):
            cs += [self.metrics.sample(r) for r in scope]
            bs += [self.metrics.sample(r) for r in base]
            (sleep or time.sleep)(gap)
        return self.decide(canary_id, _mean(cs), _mean(bs))

    def decide(self, canary_id, canary_metrics, baseline_metrics):
        c = self.store.canary(canary_id)
        passed, findings = evaluate(c["gates"], canary_metrics, baseline_metrics)
        self.store.clear_overrides(canary_id)
        if passed:
            self.store.promote(self.group, c["candidate_version_id"])
        self.store.close_canary(canary_id, PASS if passed else FAIL, "\n".join(findings))
        return (PASS if passed else FAIL), findings

    def abort(self, canary_id, reason="aborted by operator"):
        self.store.clear_overrides(canary_id)
        self.store.close_canary(canary_id, FAIL, reason)
