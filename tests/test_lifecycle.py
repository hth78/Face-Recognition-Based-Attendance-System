from fras.lifecycle import EmbeddingIndex, EmbeddingStore, _serialize_vector, train_embeddings


def _blob(values):
    return _serialize_vector(values)


def test_enrollment_to_recognition_consistency(tmp_path):
    db = tmp_path / "embeddings.sqlite"
    store = EmbeddingStore(db)

    alice = [1.0, 0.0, 0.0]
    bob = [0.0, 1.0, 0.0]

    store.collect_enrollment_sample("alice", _blob(alice))
    store.collect_enrollment_sample("bob", _blob(bob))
    train_embeddings(store)

    index = EmbeddingIndex.from_store(store)
    result_alice = index.recognize(alice, threshold=0.2)
    result_bob = index.recognize(bob, threshold=0.2)

    assert result_alice.identity == "alice"
    assert result_bob.identity == "bob"


def test_threshold_behavior(tmp_path):
    db = tmp_path / "embeddings.sqlite"
    store = EmbeddingStore(db)

    anchor = [1.0, 1.0]
    store.collect_enrollment_sample("anchor", _blob(anchor))
    train_embeddings(store)

    query = [1.0, 0.85]
    index = EmbeddingIndex.from_store(store)

    loose = index.recognize(query, threshold=0.2)
    strict = index.recognize(query, threshold=0.01)

    assert loose.identity == "anchor"
    assert strict.identity is None
    assert strict.distance > 0.01
