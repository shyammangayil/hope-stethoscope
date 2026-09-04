"""Tests for stethoscope.storage -- ObservationEventSink implementations
(architectural-evolution spec §26 Phase 2, "Storage backend").
"""
from __future__ import annotations

from stethoscope.events import ObservationEvent
from stethoscope.storage import InMemoryObservationEventSink, SQLiteObservationEventSink


def _sample_event(experiment_id="exp-1", component_id="model/layer/0/kv_cache", layer=0) -> ObservationEvent:
    return ObservationEvent(
        experiment_id=experiment_id,
        model_id="m",
        component_id=component_id,
        component_type="KVCache",
        phase="internal_state",
        layer=layer,
        token_positions=(3,),
        timestep=None,
        tensor_metadata=None,
        statistics={"l2_norm": 1.5},
        provenance={"token_source": "generated"},
    )


# ------------------------------------------------------------- in-memory

def test_in_memory_sink_round_trips_basic_fields():
    sink = InMemoryObservationEventSink()
    sink.write(_sample_event())
    rows = sink.read_all()
    assert len(rows) == 1
    assert rows[0]["component_id"] == "model/layer/0/kv_cache"
    assert rows[0]["statistics"]["l2_norm"] == 1.5
    sink.close()  # must not raise


def test_in_memory_sink_filters_by_experiment_id():
    sink = InMemoryObservationEventSink()
    sink.write(_sample_event(experiment_id="exp-1"))
    sink.write(_sample_event(experiment_id="exp-2"))
    assert len(sink.read_all()) == 2
    assert len(sink.read_all(experiment_id="exp-1")) == 1
    assert len(sink.read_all(experiment_id="exp-nonexistent")) == 0


# ------------------------------------------------------------------ sqlite

def test_sqlite_sink_round_trips_all_fields(tmp_path):
    db_path = tmp_path / "events.db"
    sink = SQLiteObservationEventSink(db_path)
    sink.write(_sample_event())
    rows = sink.read_all()
    assert len(rows) == 1
    r = rows[0]
    assert r["experiment_id"] == "exp-1"
    assert r["model_id"] == "m"
    assert r["component_id"] == "model/layer/0/kv_cache"
    assert r["component_type"] == "KVCache"
    assert r["phase"] == "internal_state"
    assert r["layer"] == 0
    assert r["token_positions"] == [3]
    assert r["timestep"] is None
    assert r["tensor_metadata"] is None
    assert r["statistics"] == {"l2_norm": 1.5}
    assert r["provenance"] == {"token_source": "generated"}
    sink.close()


def test_sqlite_sink_persists_across_reconnection(tmp_path):
    db_path = tmp_path / "events.db"
    sink1 = SQLiteObservationEventSink(db_path)
    sink1.write(_sample_event())
    sink1.close()

    sink2 = SQLiteObservationEventSink(db_path)  # reopen the same file
    rows = sink2.read_all()
    assert len(rows) == 1
    sink2.close()


def test_sqlite_sink_write_many_batches_correctly(tmp_path):
    sink = SQLiteObservationEventSink(tmp_path / "events.db")
    events = [_sample_event(component_id=f"model/layer/{i}/kv_cache", layer=i) for i in range(5)]
    sink.write_many(events)
    rows = sink.read_all()
    assert len(rows) == 5
    assert sorted(r["layer"] for r in rows) == [0, 1, 2, 3, 4]
    sink.close()


def test_sqlite_sink_write_many_with_empty_list_is_a_no_op(tmp_path):
    sink = SQLiteObservationEventSink(tmp_path / "events.db")
    sink.write_many([])  # must not raise
    assert sink.read_all() == []
    sink.close()


def test_sqlite_sink_read_all_without_filter_returns_every_experiment(tmp_path):
    # The documented footgun this sink's read_all() docstring warns
    # about -- confirmed here, not just asserted in prose.
    db_path = tmp_path / "shared.db"
    sink = SQLiteObservationEventSink(db_path)
    sink.write(_sample_event(experiment_id="exp-1"))
    sink.close()

    sink2 = SQLiteObservationEventSink(db_path)  # a second "run" against the same shared file
    sink2.write(_sample_event(experiment_id="exp-2"))
    assert len(sink2.read_all()) == 2  # both experiments, unless scoped
    assert len(sink2.read_all(experiment_id="exp-2")) == 1
    sink2.close()


def test_sqlite_sink_distinct_experiment_ids(tmp_path):
    sink = SQLiteObservationEventSink(tmp_path / "events.db")
    sink.write(_sample_event(experiment_id="exp-a"))
    sink.write(_sample_event(experiment_id="exp-b"))
    sink.write(_sample_event(experiment_id="exp-a"))
    assert sink.distinct_experiment_ids() == ["exp-a", "exp-b"]
    sink.close()
