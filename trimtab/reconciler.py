"""Reconciler. The hot path of trimtabd.

One replica, one adapter, one loop. Each tick reads the group's desired
version from the store, compares it to the version this replica last applied,
and if they differ, validates the fields against the manifest and pushes the
hot ones through the adapter. Applying the same version twice is a no-op, so
retries and restarts converge (idempotent by construction).

Cold fields in a desired version are reported as `needs_reinit` in the replica
status rather than applied. The cold path lands with the supervisor.
"""
import time

from .adapters import ApplyResult
from .manifest import Manifest
from .store import Store

HEALTHY, APPLYING, FAILED, NEEDS_REINIT = "healthy", "applying", "failed", "needs_reinit"


class Reconciler:
    def __init__(self, store: Store, manifest: Manifest, adapter, replica_id, group):
        self.store = store
        self.manifest = manifest
        self.adapter = adapter
        self.replica_id = replica_id
        self.group = group
        self.applied_version_id = None
        self.last_result: ApplyResult | None = None

    def tick(self) -> str:
        desired = self.store.desired_for(self.replica_id, self.group)
        if desired is None or desired.id == self.applied_version_id:
            self.store.record_status(self.replica_id, self.group, self.applied_version_id, HEALTHY)
            return HEALTHY

        hot, rejected = self.manifest.validate(desired.fields)
        cold = {k: r for k, r in rejected.items() if "cold" in r or "warm" in r}
        bad = {k: r for k, r in rejected.items() if k not in cold}
        if bad:
            self.store.record_status(self.replica_id, self.group, self.applied_version_id, FAILED, f"manifest rejected {bad}")
            return FAILED

        self.store.record_status(self.replica_id, self.group, self.applied_version_id, APPLYING)
        result = self.adapter.set_hot(hot) if hot else ApplyResult(True)
        self.last_result = result
        if not result.ok:
            self.store.record_status(self.replica_id, self.group, self.applied_version_id, FAILED, f"engine rejected {result.rejected}")
            return FAILED

        self.applied_version_id = desired.id
        if cold:
            self.store.record_status(self.replica_id, self.group, desired.id, NEEDS_REINIT, f"cold fields pending {sorted(cold)}")
            return NEEDS_REINIT
        self.store.record_status(self.replica_id, self.group, desired.id, HEALTHY)
        return HEALTHY

    def run(self, interval_s=1.0, stop=None):
        while stop is None or not stop.is_set():
            self.tick()
            time.sleep(interval_s)
