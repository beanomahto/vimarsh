from __future__ import annotations

import json
from app.database import get_db


def create_chat(title: str = "New Chat") -> int:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO chats (title) VALUES (%s) RETURNING id",
            (title,),
        )
        chat_id = cur.fetchone()["id"]
        return chat_id


def list_chats() -> list[dict]:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, title, updated_at FROM chats ORDER BY updated_at DESC"
        )
        rows = cur.fetchall()
        return [
            {"id": r["id"], "title": r["title"], "updated_at": str(r["updated_at"])}
            for r in rows
        ]


def get_messages(chat_id: int) -> list[dict]:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT role, content, sources FROM messages WHERE chat_id = %s ORDER BY id",
            (chat_id,),
        )
        rows = cur.fetchall()
        result = []
        for row in rows:
            msg = {"role": row["role"], "content": row["content"]}
            if row["sources"]:
                msg["sources"] = (
                    row["sources"]
                    if isinstance(row["sources"], list)
                    else json.loads(row["sources"])
                )
            result.append(msg)
        return result


def add_message(chat_id: int, role: str, content: str, sources: list | None = None):
    with get_db() as conn:
        cur = conn.cursor()
        sources_json = json.dumps(sources) if sources else None
        cur.execute(
            "INSERT INTO messages (chat_id, role, content, sources) VALUES (%s, %s, %s, %s)",
            (chat_id, role, content, sources_json),
        )
        cur.execute(
            "UPDATE chats SET updated_at = NOW() WHERE id = %s",
            (chat_id,),
        )


def update_title(chat_id: int, title: str):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE chats SET title = %s WHERE id = %s",
            (title, chat_id),
        )


def delete_chat(chat_id: int):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM chats WHERE id = %s", (chat_id,))
