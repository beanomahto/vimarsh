from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from openai import OpenAI
from pypdf import PdfReader

from app.config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    PROVIDERS,
    DEFAULT_PROVIDER,
    PARENT_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
    TOP_K_VECTOR,
    TOP_K_FTS,
    TOP_K_FINAL,
    MAX_HISTORY_TURNS,
)
from app.database import get_db

logger = logging.getLogger(__name__)

# ── LLM ──────────────────────────────────────────
_current_provider: str = DEFAULT_PROVIDER
_current_model: str = PROVIDERS[DEFAULT_PROVIDER]["default_model"]
_llm_clients: dict[str, OpenAI] = {}

# ── Embedding client (always OpenAI) ─────────────
_embed_client: OpenAI | None = None


def _get_embed_client() -> OpenAI:
    global _embed_client
    if _embed_client is None:
        _embed_client = OpenAI(api_key=OPENAI_API_KEY)
    return _embed_client


def _get_llm() -> tuple[OpenAI, str]:
    if _current_provider not in _llm_clients:
        cfg = PROVIDERS[_current_provider]
        _llm_clients[_current_provider] = OpenAI(
            api_key=cfg["api_key"],
            base_url=cfg["base_url"],
        )
    return _llm_clients[_current_provider], _current_model


def set_model(provider: str, model: str):
    global _current_provider, _current_model
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")
    if model not in PROVIDERS[provider]["models"]:
        raise ValueError(f"Unknown model: {model}")
    _current_provider = provider
    _current_model = model


def get_current_model() -> dict:
    return {"provider": _current_provider, "model": _current_model}


# ── Embeddings via OpenAI API (NO local model!) ─
def _get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings for a batch of texts using OpenAI API."""
    client = _get_embed_client()
    # OpenAI supports up to 2048 inputs per call
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in resp.data]


def _get_embedding(text: str) -> list[float]:
    """Get embedding for a single text."""
    return _get_embeddings([text])[0]


# ── Text Extraction ─────────────────────────────
def _extract_text(file_path: str) -> list[dict]:
    path = Path(file_path)
    ext = path.suffix.lower()
    source = path.name

    if ext == ".pdf":
        reader = PdfReader(file_path)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"text": text, "source": source, "page": i + 1})
        return pages
    elif ext in (".txt", ".md"):
        text = path.read_text(encoding="utf-8")
        return [{"text": text, "source": source, "page": None}]
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ── Chunking ─────────────────────────────────────
def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks


def _chunk_document(pages: list[dict]) -> tuple[list[dict], list[dict]]:
    parents = []
    children = []

    for page_data in pages:
        text = page_data["text"]
        source = page_data["source"]
        page = page_data["page"]

        parent_chunks = _split_text(text, PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP)

        for pi, parent_text in enumerate(parent_chunks):
            parent_id = hashlib.md5(
                f"{source}:{page}:{pi}:{parent_text[:100]}".encode()
            ).hexdigest()

            parents.append({
                "id": parent_id,
                "text": parent_text,
                "source": source,
                "page": page,
            })

            child_chunks = _split_text(
                parent_text, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP
            )
            for ci, child_text in enumerate(child_chunks):
                child_id = f"{parent_id}_c{ci}"
                children.append({
                    "id": child_id,
                    "parent_id": parent_id,
                    "text": child_text,
                    "source": source,
                    "page": page,
                })

    return parents, children


# ── Ingest ───────────────────────────────────────
def ingest_file(file_path: str) -> int:
    """Extract, chunk, embed, and store everything in PostgreSQL."""
    pages = _extract_text(file_path)
    parents, children = _chunk_document(pages)

    if not children:
        return 0

    # Compute embeddings via OpenAI API (batched)
    child_texts = [c["text"] for c in children]
    all_embeddings: list[list[float]] = []
    batch_size = 100
    for i in range(0, len(child_texts), batch_size):
        batch = child_texts[i : i + batch_size]
        all_embeddings.extend(_get_embeddings(batch))

    # Store everything in PostgreSQL (persistent!)
    with get_db() as conn:
        cur = conn.cursor()

        for p in parents:
            cur.execute(
                """INSERT INTO parent_chunks (id, text, source, page)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (id) DO UPDATE
                   SET text = EXCLUDED.text""",
                (p["id"], p["text"], p["source"], p["page"]),
            )

        for c, emb in zip(children, all_embeddings):
            cur.execute(
                """INSERT INTO child_chunks (id, parent_id, text, source, page, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s::vector)
                   ON CONFLICT (id) DO UPDATE
                   SET text = EXCLUDED.text, embedding = EXCLUDED.embedding""",
                (
                    c["id"],
                    c["parent_id"],
                    c["text"],
                    c["source"],
                    c["page"],
                    str(emb),
                ),
            )

    logger.info(
        f"Ingested {len(parents)} parents, {len(children)} children from {file_path}"
    )
    return len(children)


# ── Hybrid Search (pgvector + full-text search) ─
def _rrf_fuse(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def _hybrid_search(search_query: str, top_k: int = TOP_K_FINAL) -> list[dict]:
    """
    Hybrid search: pgvector cosine similarity + PostgreSQL full-text search,
    fused with RRF, then LLM-reranked.
    """
    query_embedding = _get_embedding(search_query)

    with get_db() as conn:
        cur = conn.cursor()

        # ── 1. Vector search (pgvector) ──────────
        cur.execute(
            """SELECT id, parent_id, text, source, page
               FROM child_chunks
               WHERE embedding IS NOT NULL
               ORDER BY embedding <=> %s::vector
               LIMIT %s""",
            (str(query_embedding), TOP_K_VECTOR),
        )
        vector_results = cur.fetchall()

        # ── 2. Full-text search (tsvector) ───────
        cur.execute(
            """SELECT id, parent_id, text, source, page,
                      ts_rank_cd(tsv, query) AS rank
               FROM child_chunks, plainto_tsquery('english', %s) query
               WHERE tsv @@ query
               ORDER BY rank DESC
               LIMIT %s""",
            (search_query, TOP_K_FTS),
        )
        fts_results = cur.fetchall()

    # ── 3. RRF Fusion ────────────────────────────
    vector_ids = [r["id"] for r in vector_results]
    fts_ids = [r["id"] for r in fts_results]
    fused_ids = _rrf_fuse([vector_ids, fts_ids])

    # Build lookup of all results
    all_results: dict[str, dict] = {}
    for r in vector_results:
        all_results[r["id"]] = dict(r)
    for r in fts_results:
        if r["id"] not in all_results:
            all_results[r["id"]] = dict(r)

    candidates = []
    for fid in fused_ids:
        if fid in all_results:
            candidates.append(all_results[fid])
        if len(candidates) >= top_k * 2:
            break

    if not candidates:
        return []

    # ── 4. LLM Reranking ────────────────────────
    docs_text = "\n\n".join(
        f"[DOC {i}]: {c['text'][:500]}" for i, c in enumerate(candidates)
    )
    llm, model = _get_llm()
    resp = llm.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a relevance ranker. Given a query and documents, "
                    "return the indices of the most relevant documents in order "
                    "of relevance. Return ONLY comma-separated indices "
                    "(e.g. '3,0,5,1'), nothing else."
                ),
            },
            {
                "role": "user",
                "content": f"Query: {search_query}\n\nDocuments:\n{docs_text}",
            },
        ],
        temperature=0,
        max_tokens=100,
    )

    try:
        indices = [
            int(x.strip())
            for x in resp.choices[0].message.content.strip().split(",")
            if x.strip().isdigit()
        ]
        reranked = [candidates[i] for i in indices if i < len(candidates)]
        return reranked[:top_k]
    except (ValueError, IndexError):
        return candidates[:top_k]


# ── Query Reformulation ─────────────────────────
def _reformulate_query(question: str, history: list[dict]) -> str:
    if not history:
        return question

    recent = history[-(MAX_HISTORY_TURNS * 2) :]
    history_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in recent
    )

    llm, model = _get_llm()
    resp = llm.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Rewrite the user's question to be self-contained, "
                    "resolving any pronouns or references from the chat history. "
                    "Return ONLY the rewritten question, nothing else."
                ),
            },
            {
                "role": "user",
                "content": f"Chat history:\n{history_text}\n\nQuestion: {question}",
            },
        ],
        temperature=0,
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()


# ── Get parent chunk from DB (no in-memory store!) ─
def _get_parent(parent_id: str) -> dict | None:
    """Fetch a single parent chunk from PostgreSQL."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, text, source, page FROM parent_chunks WHERE id = %s",
            (parent_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None


# ── Main Query (streaming) ──────────────────────
def query(question: str, history: list[dict] | None = None):
    search_query = _reformulate_query(question, history or [])
    results = _hybrid_search(search_query)

    seen_parents = set()
    context_parts = []
    sources = []

    for child in results:
        parent_id = child.get("parent_id", "")
        parent = _get_parent(parent_id)  # ← fetch from DB, not memory

        if parent and parent_id not in seen_parents:
            seen_parents.add(parent_id)
            context_parts.append(parent["text"])
            page = parent.get("page")
            sources.append({
                "content": parent["text"][:300],
                "source": parent.get("source", "unknown"),
                "page": page if page and page != -1 else None,
            })
        elif not parent:
            context_parts.append(child["text"])
            page = child.get("page")
            sources.append({
                "content": child["text"][:300],
                "source": child.get("source", "unknown"),
                "page": page if page and page != -1 else None,
            })

    context = "\n\n---\n\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the question based on "
                "the provided context. If the answer is not in the context, "
                "say so. Be concise and accurate."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        },
    ]

    llm, model = _get_llm()
    stream = llm.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        stream=True,
    )

    def generate():
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    return generate, sources


def query_sync(question: str, history: list[dict] | None = None) -> dict:
    generate, sources = query(question, history)
    answer = "".join(generate())
    return {"answer": answer, "sources": sources}


# ── Stats (from DB, not memory) ──────────────────
def get_stats() -> dict:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) AS count FROM child_chunks")
        child_count = cur.fetchone()["count"]
        cur.execute("SELECT COUNT(*) AS count FROM parent_chunks")
        parent_count = cur.fetchone()["count"]
    return {
        "total_chunks": child_count,
        "parent_chunks": parent_count,
        "child_chunks": child_count,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NO _rebuild_from_db() needed!
# Everything is already in PostgreSQL. Zero cold-start cost.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
