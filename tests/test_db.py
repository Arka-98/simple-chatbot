"""The `threads` table powers the sidebar, so its lifecycle is worth pinning down."""

from __future__ import annotations

import time

import pytest

from chatbot import db


@pytest.fixture()
def temp_db(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CHATBOT_DB_PATH", str(tmp_path / "test.db"))
    db.init_db()


def test_thread_is_created_on_first_message_and_keeps_its_title(temp_db) -> None:
    db.ensure_thread("t1", "Hello there")
    db.ensure_thread("t1", "A completely different second message")

    threads = db.list_threads()
    assert [thread.id for thread in threads] == ["t1"]
    assert threads[0].title == "Hello there"
    assert threads[0].updated_at >= threads[0].created_at


def test_titles_are_single_line_and_truncated(temp_db) -> None:
    db.ensure_thread("t1", "  a very long first line\n\n" + "x" * 100)

    title = db.list_threads()[0].title
    assert "\n" not in title
    assert len(title) == db.TITLE_MAX_LENGTH + 1
    assert title.endswith("…")


def test_list_threads_is_newest_first(temp_db) -> None:
    db.ensure_thread("older", "first message")
    time.sleep(0.01)
    db.ensure_thread("newer", "second message")

    assert [thread.id for thread in db.list_threads()] == ["newer", "older"]


def test_delete_thread_leaves_other_threads_alone(temp_db) -> None:
    db.ensure_thread("keep", "keep me")
    db.ensure_thread("drop", "drop me")

    db.delete_thread("drop")

    assert [thread.id for thread in db.list_threads()] == ["keep"]
