from __future__ import annotations

import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager

from app.config import DATABASE_URL, EMBEDDING_DIMENSIONS

logger = logging.getLogger(__name__)


def _get_conn():
    """Get a new database connection."""
    return psycopg2.connect(
        DATABASE_URL,
        sslmode="require",
        cursor_factory=RealDictCursor,
    )


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
    """Create tables + pgvector extension, migrate existing tables."""
    with get_db() as conn:
        cur = conn.cursor()

        # ── Enable pgvector extension ────────────
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.warning(
                f"Could not create vector extension: {e}. "
                "Enable it from Supabase Dashboard → Extensions."
            )

        # ── Chat history tables ──────────────────
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

        # ── RAG data tables ──────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS parent_chunks (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                page INTEGER
            )
        """)

        # Create child_chunks if it doesn't exist (fresh install)
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

        # ── MIGRATE: Add new columns to existing table ──
        # These are safe to run even if columns already exist

        try:
            cur.execute(f"""
                ALTER TABLE child_chunks
                ADD COLUMN IF NOT EXISTS embedding vector({EMBEDDING_DIMENSIONS})
            """)
            conn.commit()
            logger.info("Column 'embedding' ready.")
        except Exception as e:
            conn.rollback()
            logger.warning(f"Could not add embedding column: {e}")

        try:
            cur.execute("""
                ALTER TABLE child_chunks
                ADD COLUMN IF NOT EXISTS tsv tsvector
                GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
            """)
            conn.commit()
            logger.info("Column 'tsv' ready.")
        except Exception as e:
            conn.rollback()
            logger.warning(f"Could not add tsv column: {e}")

        # ── Indexes (only if columns exist) ──────
        try:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS child_chunks_hnsw_idx
                ON child_chunks
                USING hnsw (embedding vector_cosine_ops)
            """)
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.warning(f"Could not create embedding index: {e}")

        try:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS child_chunks_tsv_idx
                ON child_chunks
                USING gin (tsv)
            """)
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.warning(f"Could not create tsv index: {e}")

        logger.info("Database initialized successfully.")


# Initialize tables on import
init_db()
