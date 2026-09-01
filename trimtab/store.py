"""Append-only config store. SQLite for dev and single node, same schema as
the Postgres deployment later.

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
  id integer primary key autoincrement,
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
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(SCHEMA)

    def propose(self, group, fields, created_by, reason) -> ConfigVersion:
        """Append a new version whose parent is the group's current desired
        version. Does not change desired state. Promote does."""
        parent = self.desired(group)
        cur = self.db.execute(
            "insert into config_version(replica_group,parent_id,fields,created_by,reason,created_at) values(?,?,?,?,?,?)",
            (group, parent.id if parent else None, json.dumps(fields, sort_keys=True), created_by, reason, time.time()),
        )
        self.db.commit()
        return self.version(cur.lastrowid)

    def promote(self, group, version_id):
        v = self.version(version_id)
        if v.replica_group != group:
            raise ValueError(f"version {version_id} belongs to group {v.replica_group!r}, not {group!r}")
        self.db.execute(
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
        row = self.db.execute("select version_id from desired_state where replica_group=?", (group,)).fetchone()
        return self.version(row[0]) if row else None

    def version(self, version_id) -> ConfigVersion:
        row = self.db.execute("select * from config_version where id=?", (version_id,)).fetchone()
        if row is None:
            raise KeyError(f"no config_version {version_id}")
        return ConfigVersion(row[0], row[1], row[2], json.loads(row[3]), row[4], row[5], row[6])

    def versions(self, group) -> list[ConfigVersion]:
        rows = self.db.execute("select id from config_version where replica_group=? order by id", (group,)).fetchall()
        return [self.version(r[0]) for r in rows]

    def record_status(self, replica_id, group, applied_version_id, state, detail=""):
        self.db.execute(
            "insert into replica_status(replica_id,replica_group,applied_version_id,state,detail,last_heartbeat) values(?,?,?,?,?,?) "
            "on conflict(replica_id) do update set applied_version_id=excluded.applied_version_id, state=excluded.state, "
            "detail=excluded.detail, last_heartbeat=excluded.last_heartbeat",
            (replica_id, group, applied_version_id, state, detail, time.time()),
        )
        self.db.commit()

    def status(self, group) -> list[dict]:
        rows = self.db.execute(
            "select replica_id,applied_version_id,state,detail,last_heartbeat from replica_status where replica_group=? order by replica_id",
            (group,),
        ).fetchall()
        return [dict(replica_id=r[0], applied_version_id=r[1], state=r[2], detail=r[3], last_heartbeat=r[4]) for r in rows]
