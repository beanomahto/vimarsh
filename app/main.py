import json
import os
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import DOCUMENTS_DIR, PROVIDERS
from app.rag import ingest_file, query, query_sync, get_stats, set_model, get_current_model

app = FastAPI(title="RAG Chatbot API")

os.makedirs(DOCUMENTS_DIR, exist_ok=True)


class QueryRequest(BaseModel):
    question: str
    history: list[dict] = []
    stream: bool = True


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in (".pdf", ".txt", ".md"):
        raise HTTPException(400, f"Unsupported file type: {ext}")

    file_path = os.path.join(DOCUMENTS_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks = ingest_file(file_path)
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
