from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from openai import OpenAI
from rank_bm25 import BM25Okapi
from pypdf import PdfReader

from app.config import (
    PROVIDERS,
    DEFAULT_PROVIDER,
    CHROMA_PERSIST_DIR,
    PARENT_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
    TOP_K_VECTOR,
    TOP_K_BM25,
    TOP_K_FINAL,
    MAX_HISTORY_TURNS,
)

_current_provider: str = DEFAULT_PROVIDER
_current_model: str = PROVIDERS[DEFAULT_PROVIDER]["default_model"]
_llm_clients: dict[str, OpenAI] = {}
_embed_fn = DefaultEmbeddingFunction()


def _get_llm() -> tuple[OpenAI, str]:
    if _current_provider not in _llm_clients:
        cfg = PROVIDERS[_current_provider]
        _llm_clients[_current_provider] = OpenAI(
            api_key=cfg["api_key"], base_url=cfg["base_url"],
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
_chroma = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
_child_collection = _chroma.get_or_create_collection(
    name="child_chunks_v2",
    metadata={"hnsw:space": "cosine"},
    embedding_function=_embed_fn,
)
_parent_store: dict[str, dict] = {}
_bm25_corpus: list[dict] = []
_bm25_index: BM25Okapi | None = None



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

            child_chunks = _split_text(parent_text, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP)
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



def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _rebuild_bm25():
    global _bm25_index, _bm25_corpus
    results = _child_collection.get(include=["documents", "metadatas"])
    _bm25_corpus = []
    for doc_id, doc_text, meta in zip(
        results["ids"], results["documents"], results["metadatas"]
    ):
        _bm25_corpus.append({
            "id": doc_id,
            "text": doc_text,
            "parent_id": meta.get("parent_id", ""),
            "source": meta.get("source", ""),
            "page": meta.get("page"),
        })

    if _bm25_corpus:
        tokenized = [_tokenize(c["text"]) for c in _bm25_corpus]
        _bm25_index = BM25Okapi(tokenized)
    else:
        _bm25_index = None



def ingest_file(file_path: str) -> int:
    pages = _extract_text(file_path)
    parents, children = _chunk_document(pages)

    for p in parents:
        _parent_store[p["id"]] = p

    batch_size = 100
    for i in range(0, len(children), batch_size):
        batch = children[i : i + batch_size]
        texts = [c["text"] for c in batch]
        ids = [c["id"] for c in batch]
        metadatas = [
            {
                "parent_id": c["parent_id"],
                "source": c["source"],
                "page": c["page"] if c["page"] is not None else -1,
            }
            for c in batch
        ]
        _child_collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )

    _rebuild_bm25()
    return len(children)



def _rrf_fuse(
    ranked_lists: list[list[str]], k: int = 60
) -> list[str]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    sorted_ids = sorted(scores, key=scores.get, reverse=True)
    return sorted_ids


def _hybrid_search(query: str, top_k: int = TOP_K_FINAL) -> list[dict]:
    vector_results = _child_collection.query(
        query_texts=[query],
        n_results=min(TOP_K_VECTOR, _child_collection.count() or 1),
        include=["documents", "metadatas"],
    )
    vector_ids = vector_results["ids"][0] if vector_results["ids"] else []

    bm25_ids = []
    if _bm25_index is not None and _bm25_corpus:
        tokenized_query = _tokenize(query)
        bm25_scores = _bm25_index.get_scores(tokenized_query)
        scored = list(zip(range(len(_bm25_corpus)), bm25_scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        bm25_ids = [_bm25_corpus[idx]["id"] for idx, _ in scored[:TOP_K_BM25]]

    fused_ids = _rrf_fuse([vector_ids, bm25_ids])

    all_results = {}
    if vector_results["ids"] and vector_results["ids"][0]:
        for i, vid in enumerate(vector_results["ids"][0]):
            all_results[vid] = {
                "id": vid,
                "text": vector_results["documents"][0][i],
                "parent_id": vector_results["metadatas"][0][i].get("parent_id", ""),
                "source": vector_results["metadatas"][0][i].get("source", ""),
                "page": vector_results["metadatas"][0][i].get("page"),
            }

    for c in _bm25_corpus:
        if c["id"] not in all_results:
            all_results[c["id"]] = c

    candidates = []
    for fid in fused_ids:
        if fid in all_results:
            candidates.append(all_results[fid])
        if len(candidates) >= top_k * 2:
            break

    if not candidates:
        return []

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
                    "return the indices of the most relevant documents in order of relevance. "
                    "Return ONLY comma-separated indices (e.g. '3,0,5,1'), nothing else."
                ),
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nDocuments:\n{docs_text}",
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



def query(question: str, history: list[dict] | None = None):
    search_query = _reformulate_query(question, history or [])

    results = _hybrid_search(search_query)

    seen_parents = set()
    context_parts = []
    sources = []

    for child in results:
        parent_id = child.get("parent_id", "")
        parent = _parent_store.get(parent_id)

        if parent and parent_id not in seen_parents:
            seen_parents.add(parent_id)
            context_parts.append(parent["text"])
            page = parent.get("page")
            sources.append({
                "content": parent["text"][:300],
                "source": parent.get("source", "unknown"),
                "page": page if page != -1 else None,
            })
        elif not parent:
            context_parts.append(child["text"])
            page = child.get("page")
            sources.append({
                "content": child["text"][:300],
                "source": child.get("source", "unknown"),
                "page": page if page != -1 else None,
            })

    context = "\n\n---\n\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the question based on the provided context. "
                "If the answer is not in the context, say so. Be concise and accurate."
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



def get_stats() -> dict:
    child_count = _child_collection.count()
    return {
        "total_chunks": child_count,
        "parent_chunks": len(_parent_store),
        "child_chunks": child_count,
    }
