"""Append-only config store. SQLite for dev and single node, Postgres for a
deployment, one class, one schema. Store(path) opens SQLite, Store(dsn) with
a postgresql:// dsn opens Postgres (needs psycopg).

config_version rows are never updated or deleted. Rolling back means pointing
desired_state at an older version id, which is itself a new event in the
log. The table is the history.
"""
import json
import sqlite3
import time
from dataclasses import dataclass

SCHEMA = """
create table if not exists config_version(
  id {pk},
  replica_group text not null,
  parent_id integer,
  fields text not null,
  created_by text not null,
  reason text not null,
  created_at real not null
);
create table if not exists desired_state(
  replica_group text primary key,
  version_id integer not null,
  updated_at real not null
);
create table if not exists replica_override(
  replica_id text primary key,
  version_id integer not null,
  canary_id integer not null
);
create table if not exists canary(
  id {pk},
  replica_group text not null,
  candidate_version_id integer not null,
  baseline_version_id integer,
  scope text not null,
  gates text not null,
  window_s real not null,
  status text not null,
  started_at real not null,
  decided_at real,
  decision text not null default ''
);
create table if not exists replica_status(
  replica_id text primary key,
  replica_group text not null,
  applied_version_id integer,
  state text not null,
  detail text not null default '',
  last_heartbeat real not null
);
"""


@dataclass(frozen=True)
class ConfigVersion:
    id: int
    replica_group: str
    parent_id: int | None
    fields: dict
    created_by: str
    reason: str
    created_at: float


class Store:
    def __init__(self, path=":memory:"):
        self.pg = str(path).startswith(("postgresql://", "postgres://"))
        if self.pg:
            import psycopg

            self.db = psycopg.connect(path, autocommit=False)
            for stmt in SCHEMA.format(pk="serial primary key").split(";"):
                if stmt.strip():
                    self.db.execute(stmt)
            self.db.commit()
        else:
            self.db = sqlite3.connect(path, check_same_thread=False)
            self.db.executescript(SCHEMA.format(pk="integer primary key autoincrement"))

    def _exec(self, sql, params=(), returning_id=False):
        """One call shape for both dialects. Placeholders are written as ? and
        translated for Postgres, and a new row id comes back the same way."""
        if self.pg:
            sql = sql.replace("?", "%s").replace("insert or replace into", "insert into")
            if returning_id:
                sql += " returning id"
            cur = self.db.execute(sql, params)
            return cur.fetchone()[0] if returning_id else cur
        cur = self.db.execute(sql, params)
        return cur.lastrowid if returning_id else cur

    def propose(self, group, fields, created_by, reason) -> ConfigVersion:
        """Append a new version whose parent is the group's current desired
        version. Does not change desired state. Promote does."""
        parent = self.desired(group)
        new_id = self._exec(
            "insert into config_version(replica_group,parent_id,fields,created_by,reason,created_at) values(?,?,?,?,?,?)",
            (group, parent.id if parent else None, json.dumps(fields, sort_keys=True), created_by, reason, time.time()),
            returning_id=True,
        )
        self.db.commit()
        return self.version(new_id)

    def promote(self, group, version_id):
        v = self.version(version_id)
        if v.replica_group != group:
            raise ValueError(f"version {version_id} belongs to group {v.replica_group!r}, not {group!r}")
        self._exec(
            "insert into desired_state(replica_group,version_id,updated_at) values(?,?,?) "
            "on conflict(replica_group) do update set version_id=excluded.version_id, updated_at=excluded.updated_at",
            (group, version_id, time.time()),
        )
        self.db.commit()

    def rollback(self, group, to_version_id):
        """Rollback is promote of an older version. Kept as its own verb so the
        intent is visible in the call site and the CLI."""
        self.promote(group, to_version_id)

    def desired(self, group) -> ConfigVersion | None:
        row = self._exec("select version_id from desired_state where replica_group=?", (group,)).fetchone()
        return self.version(row[0]) if row else None

    def desired_for(self, replica_id, group) -> ConfigVersion | None:
        """A replica's target. A canary override wins over the group's desired."""
        row = self._exec("select version_id from replica_override where replica_id=?", (replica_id,)).fetchone()
        return self.version(row[0]) if row else self.desired(group)

    def set_overrides(self, replica_ids, version_id, canary_id):
        for r in replica_ids:
            self._exec("insert or replace into replica_override(replica_id,version_id,canary_id) values(?,?,?) "
                       "on conflict(replica_id) do update set version_id=excluded.version_id, canary_id=excluded.canary_id", (r, version_id, canary_id))
        self.db.commit()

    def clear_overrides(self, canary_id):
        self._exec("delete from replica_override where canary_id=?", (canary_id,))
        self.db.commit()

    def open_canary(self, group, candidate_id, scope, gates, window_s) -> int:
        base = self.desired(group)
        new_id = self._exec(
            "insert into canary(replica_group,candidate_version_id,baseline_version_id,scope,gates,window_s,status,started_at) values(?,?,?,?,?,?,?,?)",
            (group, candidate_id, base.id if base else None, json.dumps(sorted(scope)), json.dumps(gates), window_s, "observing", time.time()),
            returning_id=True)
        self.db.commit()
        return new_id

    def canary(self, canary_id) -> dict:
        r = self._exec("select id,replica_group,candidate_version_id,baseline_version_id,scope,gates,window_s,status,started_at,decided_at,decision from canary where id=?", (canary_id,)).fetchone()
        if r is None:
            raise KeyError(f"no canary {canary_id}")
        keys = ["id","replica_group","candidate_version_id","baseline_version_id","scope","gates","window_s","status","started_at","decided_at","decision"]
        d = dict(zip(keys, r)); d["scope"] = json.loads(d["scope"]); d["gates"] = json.loads(d["gates"]); return d

    def close_canary(self, canary_id, status, decision):
        self._exec("update canary set status=?, decided_at=?, decision=? where id=?", (status, time.time(), decision, canary_id))
        self.db.commit()

    def version(self, version_id) -> ConfigVersion:
        row = self._exec("select id,replica_group,parent_id,fields,created_by,reason,created_at from config_version where id=?", (version_id,)).fetchone()
        if row is None:
            raise KeyError(f"no config_version {version_id}")
        return ConfigVersion(row[0], row[1], row[2], json.loads(row[3]), row[4], row[5], row[6])

    def versions(self, group) -> list[ConfigVersion]:
        rows = self._exec("select id from config_version where replica_group=? order by id", (group,)).fetchall()
        return [self.version(r[0]) for r in rows]

    def record_status(self, replica_id, group, applied_version_id, state, detail=""):
        self._exec(
            "insert into replica_status(replica_id,replica_group,applied_version_id,state,detail,last_heartbeat) values(?,?,?,?,?,?) "
            "on conflict(replica_id) do update set replica_group=excluded.replica_group, applied_version_id=excluded.applied_version_id, state=excluded.state, "
            "detail=excluded.detail, last_heartbeat=excluded.last_heartbeat",
            (replica_id, group, applied_version_id, state, detail, time.time()),
        )
        self.db.commit()

    def status(self, group) -> list[dict]:
        rows = self._exec(
            "select replica_id,applied_version_id,state,detail,last_heartbeat from replica_status where replica_group=? order by replica_id",
            (group,),
        ).fetchall()
        return [dict(replica_id=r[0], applied_version_id=r[1], state=r[2], detail=r[3], last_heartbeat=r[4]) for r in rows]
