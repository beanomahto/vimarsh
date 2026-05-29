# database.py

from __future__ import annotations

import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager

from app.config import DATABASE_URL


def _get_conn():
    """Get a new database connection."""
    return psycopg2.connect(DATABASE_URL, 
                            sslmode="require",cursor_factory=RealDictCursor)


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = _get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create all tables if they don't exist."""
    with get_db() as conn:
        cur = conn.cursor()

        # Chat history tables
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id SERIAL PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        # RAG data tables
        cur.execute("""
            CREATE TABLE IF NOT EXISTS parent_chunks (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                page INTEGER
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS child_chunks (
                id TEXT PRIMARY KEY,
                parent_id TEXT NOT NULL REFERENCES parent_chunks(id) ON DELETE CASCADE,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                page INTEGER
            )
        """)

        conn.commit()


# Initialize tables on import
init_db()
