#backfill_embeddings.py

"""
One-time script to backfill embeddings for existing child_chunks
that were inserted before the pgvector migration.
Usage:
    python -m scripts.backfill_embeddings
Requires HF_API_TOKEN in environment (or .env file).
"""

import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.database import get_db
from app.rag import _get_embeddings
BATCH_SIZE = 50
def backfill():
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, text FROM child_chunks WHERE embedding IS NULL")
        rows = cur.fetchall()
    if not rows:
        print("All chunks already have embeddings. Nothing to do.")
        return
    print(f"Found {len(rows)} chunks without embeddings. Backfilling...")
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        texts = [r["text"] for r in batch]
        embeddings = _get_embeddings(texts)
        with get_db() as conn:
            cur = conn.cursor()
            for row, emb in zip(batch, embeddings):
                cur.execute(
                    "UPDATE child_chunks SET embedding = %s::vector WHERE id = %s",
                    (str(emb), row["id"]),
                )
        done = min(i + BATCH_SIZE, len(rows))
        print(f"  Backfilled {done} / {len(rows)} chunks")
    print("Done!")
if __name__ == "__main__":
    backfill()