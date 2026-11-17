import sqlite3

import pytest

from d1.mapper.answer_cache import AnswerCache, cache_key


def test_put_then_get_survives_reopening(tmp_path):
    path = tmp_path / "sub" / "answers.sqlite"  # the folder is made
    cache = AnswerCache(path)
    key = cache_key("laya-v1", {"body": "Leveranciers"}, {"q0": {"criteria": ["440", "none"]}})
    assert cache.get(key) is None
    cache.put(key, {"440": 0.9, "none": 0.1})
    assert AnswerCache(path).get(key) == {"440": 0.9, "none": 0.1}
    assert len(AnswerCache(path)) == 1


def test_many_at_once(tmp_path):
    cache = AnswerCache(tmp_path / "a.sqlite")
    cache.put_many({"k1": {"x": 1}, "k2": {"x": 2}})
    assert cache.get_many(["k1", "k2", "k3"]) == {"k1": {"x": 1}, "k2": {"x": 2}}
    cache.put("k1", {"x": 10})  # a newer answer replaces the old one
    assert cache.get("k1") == {"x": 10} and len(cache) == 2
    assert cache.get_many([]) == {}


def test_key_is_stable_and_order_independent_for_dicts():
    a = cache_key("v1", {"body": "Huur", "x": 1}, ["440", "400"])
    assert a == cache_key("v1", {"x": 1, "body": "Huur"}, ["440", "400"])  # dict key order doesn't matter
    assert a != cache_key("v1", {"body": "Huur", "x": 1}, ["400", "440"])  # list order does (it's the option order)
    assert a != cache_key("v2", {"body": "Huur", "x": 1}, ["440", "400"])
    assert len(a) == 64


def test_a_damaged_file_is_an_error_that_names_it(tmp_path):
    path = tmp_path / "answers.sqlite"
    path.write_bytes(b"not a database at all" * 100)
    with pytest.raises(sqlite3.DatabaseError, match="answers.sqlite"):
        AnswerCache(path).get("k")
