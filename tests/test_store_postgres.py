"""The same store contract against a real Postgres. Skips only when no dsn is
set, so CI without Postgres still passes and a machine with it proves it."""
import os
import uuid

import pytest

from trimtab.store import Store

DSN = os.environ.get("TRIMTAB_TEST_PG_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="set TRIMTAB_TEST_PG_DSN to run")


@pytest.fixture
def store():
    s = Store(DSN)
    yield s
    s.db.close()


def test_versions_promote_rollback_status_canary_on_postgres(store):
    g = f"g-{uuid.uuid4().hex[:8]}"
    v1 = store.propose(g, {"max_running_requests": 32}, "t", "baseline")
    v2 = store.propose(g, {"max_running_requests": 8}, "t", "experiment")
    assert v2.parent_id is None  # nothing promoted yet, parent is desired at propose time
    store.promote(g, v1.id)
    v3 = store.propose(g, {"max_running_requests": 4}, "t", "third")
    assert v3.parent_id == v1.id
    assert store.desired(g).id == v1.id
    store.rollback(g, v2.id)
    assert store.desired(g).id == v2.id
    assert [v.id for v in store.versions(g)] == [v1.id, v2.id, v3.id]
    assert store.version(v1.id).fields == {"max_running_requests": 32}

    store.record_status("r0", g, v2.id, "healthy", "reinit_s=1.5")
    store.record_status("r0", g, v2.id, "healthy", "")  # upsert, same row
    assert len(store.status(g)) == 1 and store.status(g)[0]["applied_version_id"] == v2.id

    cid = store.open_canary(g, v3.id, ["r0"], [{"metric": "x", "max_abs": 1}], 30)
    store.set_overrides(["r0"], v3.id, cid)
    store.set_overrides(["r0"], v3.id, cid)  # idempotent upsert
    assert store.desired_for("r0", g).id == v3.id and store.desired_for("r1", g).id == v2.id
    store.clear_overrides(cid)
    assert store.desired_for("r0", g).id == v2.id
    store.close_canary(cid, "reverted", "x: too high")
    c = store.canary(cid)
    assert c["status"] == "reverted" and c["scope"] == ["r0"] and c["gates"][0]["metric"] == "x"
    with pytest.raises(ValueError):
        store.promote("other-group", v1.id)
