from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from app.config import BASE_DIR

DB_PATH = str(BASE_DIR / "chat_history.db")


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
        )"""
    )
    conn.execute("PRAGMA foreign_keys = ON")
    conn.commit()
    return conn


def create_chat(title: str = "New Chat") -> int:
    conn = _conn()
    now = datetime.now().isoformat()
    cur = conn.execute(
        "INSERT INTO chats (title, created_at, updated_at) VALUES (?, ?, ?)",
        (title, now, now),
    )
    conn.commit()
    chat_id = cur.lastrowid
    conn.close()
    return chat_id


def list_chats() -> list[dict]:
    conn = _conn()
    rows = conn.execute(
        "SELECT id, title, updated_at FROM chats ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1], "updated_at": r[2]} for r in rows]


def get_messages(chat_id: int) -> list[dict]:
    conn = _conn()
    rows = conn.execute(
        "SELECT role, content, sources FROM messages WHERE chat_id = ? ORDER BY id",
        (chat_id,),
    ).fetchall()
    conn.close()
    result = []
    for role, content, sources_json in rows:
        msg = {"role": role, "content": content}
        if sources_json:
            msg["sources"] = json.loads(sources_json)
        result.append(msg)
    return result


def add_message(chat_id: int, role: str, content: str, sources: list | None = None):
    conn = _conn()
    now = datetime.now().isoformat()
    sources_json = json.dumps(sources) if sources else None
    conn.execute(
        "INSERT INTO messages (chat_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)",
        (chat_id, role, content, sources_json, now),
    )
    conn.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now, chat_id))
    conn.commit()
    conn.close()


def update_title(chat_id: int, title: str):
    conn = _conn()
    conn.execute("UPDATE chats SET title = ? WHERE id = ?", (title, chat_id))
    conn.commit()
    conn.close()


def delete_chat(chat_id: int):
    conn = _conn()
    conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    conn.commit()
    conn.close()
