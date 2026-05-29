# main.py

import json
import os
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import DOCUMENTS_DIR, PROVIDERS, SUPABASE_URL, SUPABASE_KEY
from app.rag import ingest_file, query, query_sync, get_stats, set_model, get_current_model

app = FastAPI(title="RAG Chatbot API")

os.makedirs(DOCUMENTS_DIR, exist_ok=True)

# ━━━ NEW: Supabase Storage client (for PDF persistence) ━━━
_supabase_client = None

def _get_supabase():
    global _supabase_client
    if _supabase_client is None and SUPABASE_URL and SUPABASE_KEY:
        from supabase import create_client
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase_client


class QueryRequest(BaseModel):
    question: str
    history: list[dict] = []
    stream: bool = True


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in (".pdf", ".txt", ".md"):
        raise HTTPException(400, f"Unsupported file type: {ext}")

    # Read file bytes once
    file_bytes = await file.read()

    # Save locally (temp) for text extraction
    file_path = os.path.join(DOCUMENTS_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(file_bytes)

    # ━━━ NEW: Upload to Supabase Storage (persistent copy) ━━━
    sb = _get_supabase()
    if sb:
        try:
            sb.storage.from_("documents").upload(
                path=file.filename,
                file=file_bytes,
                file_options={
                    "content-type": file.content_type or "application/octet-stream",
                    "upsert": "true",
                },
            )
        except Exception as e:
            print(f"Warning: Failed to upload to Supabase Storage: {e}")

    # Ingest (extract text, chunk, index, save to DB)
    chunks = ingest_file(file_path)

    # ━━━ NEW: Clean up local temp file ━━━
    try:
        os.remove(file_path)
    except OSError:
        pass

    return {"filename": file.filename, "chunks": chunks}


@app.post("/query")
async def ask(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    if not req.stream:
        result = query_sync(req.question, req.history)
        return result

    generate, sources = query(req.question, req.history)

    def event_stream():
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
        for token in generate():
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/stats")
async def stats():
    return get_stats()


@app.get("/providers")
async def providers():
    result = {}
    for name, cfg in PROVIDERS.items():
        has_key = bool(cfg["api_key"])
        result[name] = {
            "models": cfg["models"],
            "default_model": cfg["default_model"],
            "available": has_key,
        }
    return {"providers": result, "current": get_current_model()}


class ModelRequest(BaseModel):
    provider: str
    model: str


@app.post("/model")
async def change_model(req: ModelRequest):
    try:
        set_model(req.provider, req.model)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return get_current_model()
