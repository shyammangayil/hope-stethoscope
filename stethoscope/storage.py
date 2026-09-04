"""stethoscope/storage.py
=========================
ObservationEventSink: persistence for ObservationEvent (architectural-
evolution spec §26 Phase 2, "Storage backend",
`docs/ARCHITECTURE_EVOLUTION_SPEC.md`).

Deliberately mirrors `core.storage.EventSink`'s exact shape (write/
read_all/close) and SQLite-backed design, including its hard-won
`read_all()` warning below -- the same reasoning that already proved out
there (plain stdlib `sqlite3`, zero extra dependencies, single-file,
trivially inspectable, and a real, documented footgun around unscoped
reads on a shared persistent path) applies here just as directly, and
matching an already-proven pattern beats inventing a different one for a
structurally similar problem (architecture delta report §7's "preserve"
list names this repo's SQLite-first persistence convention explicitly).

This is a SEPARATE sink from `core.storage.EventSink`, not a replacement
for it: `ObservationEvent` and `KVEvent` are different shapes
(`ObservationEvent` has three dict-valued fields --
`tensor_metadata`/`statistics`/`provenance`; `KVEvent` has one, `extra`).
See `stethoscope/observer.py` for how a `KVEvent` stream produced by the
existing, unmodified `core.hooks.CacheTap` gets bridged into
`ObservationEvent`s; this module only concerns itself with storing
`ObservationEvent`s once they exist, independent of where they came from.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, Optional, Protocol

from stethoscope.events import ObservationEvent


class ObservationEventSink(Protocol):
    def write(self, event: ObservationEvent) -> None: ...
    def read_all(self, experiment_id: Optional[str] = None) -> list[dict]: ...
    def close(self) -> None: ...


class InMemoryObservationEventSink:
    """Simplest concrete sink -- default choice for tests and short-lived
    experiments; nothing persists past process exit. `Observer` (see
    `observer.py`) uses this as its own default inner sink."""

    def __init__(self) -> None:
        self._events: list[ObservationEvent] = []

    def write(self, event: ObservationEvent) -> None:
        self._events.append(event)

    def read_all(self, experiment_id: Optional[str] = None) -> list[dict]:
        events = self._events if experiment_id is None else [
            e for e in self._events if e.experiment_id == experiment_id
        ]
        return [e.to_dict() for e in events]

    def close(self) -> None:
        pass  # nothing to release


_EVENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS observation_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    component_id TEXT NOT NULL,
    component_type TEXT NOT NULL,
    phase TEXT NOT NULL,
    layer INTEGER,
    token_positions_json TEXT,
    timestep INTEGER,
    tensor_metadata_json TEXT,
    statistics_json TEXT,
    provenance_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_obs_events_experiment ON observation_events(experiment_id);
CREATE INDEX IF NOT EXISTS idx_obs_events_component ON observation_events(component_id);
CREATE INDEX IF NOT EXISTS idx_obs_events_layer ON observation_events(layer);
"""


class SQLiteObservationEventSink:
    """Persistent sink. One `observation_events` table, append-only,
    indexed by experiment/component/layer -- the queries a
    causal-experiment report (Phase 7) or a future visualization would
    actually run. Dict-valued fields (`tensor_metadata`, `statistics`,
    `provenance`) are stored as JSON text, matching
    `core.storage.KVEvent.to_row()`'s own `extra_json` convention.

    `tensor_reference` is never persisted here -- matches
    `ObservationEvent.to_dict()`'s own default (see its docstring): it
    may hold a live, non-JSON-safe tensor object, and per spec §9 isn't
    even guaranteed to be a full tensor in the first place.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(_EVENT_SCHEMA)
        self._conn.commit()

    def write(self, event: ObservationEvent) -> None:
        self._insert_row(event.to_dict())
        self._conn.commit()

    def write_many(self, events: Iterable[ObservationEvent]) -> None:
        events = list(events)
        if not events:
            return
        for e in events:
            self._insert_row(e.to_dict())
        self._conn.commit()

    def _insert_row(self, d: dict) -> None:
        self._conn.execute(
            "INSERT INTO observation_events "
            "(experiment_id, model_id, component_id, component_type, phase, layer, "
            " token_positions_json, timestep, tensor_metadata_json, statistics_json, provenance_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                d["experiment_id"], d["model_id"], d["component_id"], d["component_type"], d["phase"],
                d["layer"], json.dumps(d["token_positions"]), d["timestep"],
                json.dumps(d["tensor_metadata"]), json.dumps(d["statistics"]), json.dumps(d["provenance"]),
            ),
        )

    def read_all(self, experiment_id: Optional[str] = None) -> list[dict]:
        """Without `experiment_id`, returns every row ever written to this
        file -- which, for a persistent path shared across multiple runs,
        means rows from EVERY past experiment, not just the current one.
        Pass `experiment_id` (`Observer` generates a fresh one per
        instance unless you supply your own) to scope to one experiment.
        Not a hypothetical: `core.storage.SQLiteEventSink.read_all()`'s
        own docstring documents this exact footgun having actually caused
        confusion once already in this project (see CHANGELOG.md) -- the
        same risk applies identically here, for the same reason."""
        if experiment_id is not None:
            cur = self._conn.execute(
                "SELECT * FROM observation_events WHERE experiment_id = ? ORDER BY id", (experiment_id,)
            )
        else:
            cur = self._conn.execute("SELECT * FROM observation_events ORDER BY id")
        cols = [c[0] for c in cur.description]
        rows = []
        for raw in cur.fetchall():
            d = dict(zip(cols, raw))
            d["token_positions"] = json.loads(d.pop("token_positions_json"))
            d["tensor_metadata"] = json.loads(d.pop("tensor_metadata_json"))
            d["statistics"] = json.loads(d.pop("statistics_json"))
            d["provenance"] = json.loads(d.pop("provenance_json"))
            rows.append(d)
        return rows

    def distinct_experiment_ids(self) -> list[str]:
        """All experiment_ids present in this file -- the fastest way to
        notice "oh, this file has 6 old experiments in it" before it
        bites you, matching `core.storage.SQLiteEventSink.distinct_run_ids()`."""
        cur = self._conn.execute("SELECT DISTINCT experiment_id FROM observation_events ORDER BY experiment_id")
        return [row[0] for row in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()
