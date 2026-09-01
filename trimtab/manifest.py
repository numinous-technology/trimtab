"""Knob manifest. The registry that classifies every engine parameter.

A manifest is YAML shipped per engine version under engine/<engine>/manifests.
Loading it yields a Manifest whose validate() runs at the control plane
boundary, before anything reaches an adapter. Cold knobs are rejected here with
a message that says they need a reinit. Ceilings sourced from the boot value
are enforced by the engine, not here, because only the engine knows them.
"""
import re
from dataclasses import dataclass
from pathlib import Path

HOT, COLD, DEFERRED = "hot", "cold", "deferred"


@dataclass(frozen=True)
class Knob:
    field: str
    klass: str
    vmin: int | None = None
    vmax: int | None = None
    max_source: str | None = None
    vtype: str = "int"


class Manifest:
    def __init__(self, engine, engine_commit, knobs):
        self.engine = engine
        self.engine_commit = engine_commit
        self.knobs = {k.field: k for k in knobs}

    def validate(self, changes: dict) -> tuple[dict, dict]:
        accepted, rejected = {}, {}
        for k, v in changes.items():
            knob = self.knobs.get(k)
            if knob is None:
                rejected[k] = "not in manifest"
            elif knob.klass == COLD:
                rejected[k] = "cold knob, needs a reinit"
            elif knob.klass == DEFERRED:
                rejected[k] = "not hot in this engine version"
            elif isinstance(v, bool) or not isinstance(v, int if knob.vtype == "int" else (int, float)):
                rejected[k] = f"must be {'an integer' if knob.vtype == 'int' else 'a number'}"
            elif knob.vmin is not None and v < knob.vmin:
                rejected[k] = f"below minimum {knob.vmin}"
            elif knob.vmax is not None and v > knob.vmax:
                rejected[k] = f"above maximum {knob.vmax}"
            else:
                accepted[k] = v
        return accepted, rejected

    def hot_fields(self):
        return sorted(k for k, kn in self.knobs.items() if kn.klass == HOT)


def _parse_yaml(text):
    """Minimal parser for the manifest subset we write. No PyYAML dependency.
    Handles top-level scalars and a `knobs:` list of flat mappings with one
    optional nested `validator:` mapping."""
    top, knobs, cur, in_validator = {}, [], None, False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        body = line.strip()
        if indent == 0:
            in_validator = False
            if body == "knobs:":
                continue
            key, _, val = body.partition(":")
            top[key.strip()] = val.strip().strip('"')
        elif body.startswith("- "):
            in_validator = False
            cur = {}
            knobs.append(cur)
            key, _, val = body[2:].partition(":")
            cur[key.strip()] = val.strip().strip('"')
        elif body == "validator:":
            in_validator = True
            cur["validator"] = {}
        else:
            key, _, val = body.partition(":")
            target = cur["validator"] if in_validator and indent >= 6 else cur
            if not (in_validator and indent >= 6):
                in_validator = False
            target[key.strip()] = val.strip().strip('"')
    return top, knobs


def load(path) -> Manifest:
    top, raw_knobs = _parse_yaml(Path(path).read_text())
    knobs = []
    for r in raw_knobs:
        v = r.get("validator", {})
        knobs.append(Knob(
            field=r["field"],
            klass=r["klass"],
            vmin=int(v["min"]) if "min" in v else None,
            vmax=int(v["max"]) if "max" in v else None,
            max_source=v.get("max_source"),
            vtype=v.get("type", "int"),
        ))
    return Manifest(top["engine"], top["engine_commit"], knobs)


def find(engine: str, root=None) -> Manifest:
    """Load the newest manifest for an engine from the repo layout."""
    root = Path(root or Path(__file__).resolve().parent.parent)
    files = sorted((root / "engine" / engine / "manifests").glob("*.yaml"))
    if not files:
        raise FileNotFoundError(f"no manifest for {engine}")
    return load(files[-1])
