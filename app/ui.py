import json

import requests
import streamlit as st

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chat_history import (
    create_chat, list_chats, get_messages, add_message,
    update_title, delete_chat,
)

# API_URL = "http://localhost:8000"
API_URL = "https://your-backend-url.onrender.com"


st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("RAG Chatbot")
st.caption("Hybrid search (BM25 + vector) with re-ranking and streaming")

if "chat_id" not in st.session_state:
    st.session_state.chat_id = None

with st.sidebar:
    if st.button("New Chat", use_container_width=True, type="primary"):
        chat_id = create_chat()
        st.session_state.chat_id = chat_id
        st.rerun()

    st.divider()
    st.subheader("History")
    chats = list_chats()
    for chat in chats:
        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(
                chat["title"],
                key=f"chat_{chat['id']}",
                use_container_width=True,
            ):
                st.session_state.chat_id = chat["id"]
                st.rerun()
        with col2:
            if st.button("X", key=f"del_{chat['id']}"):
                delete_chat(chat["id"])
                if st.session_state.chat_id == chat["id"]:
                    st.session_state.chat_id = None
                st.rerun()

    st.divider()
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop PDF, TXT, or MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                resp = requests.post(
                    f"{API_URL}/ingest",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"{uploaded_file.name}: {data['chunks']} chunks indexed")
                else:
                    st.error(f"Failed to process {uploaded_file.name}")

    st.divider()
    st.subheader("Model")
    try:
        prov_data = requests.get(f"{API_URL}/providers", timeout=3).json()
        providers = prov_data["providers"]
        current = prov_data["current"]

        available = [p for p, cfg in providers.items() if cfg["available"]]
        provider = st.selectbox(
            "Provider",
            available,
            index=available.index(current["provider"]) if current["provider"] in available else 0,
            key="provider_select",
        )

        models = providers[provider]["models"]
        cur_model = current["model"] if current["provider"] == provider else providers[provider]["default_model"]
        model = st.selectbox(
            "Model",
            models,
            index=models.index(cur_model) if cur_model in models else 0,
            key="model_select",
        )

        if provider != current["provider"] or model != current["model"]:
            requests.post(
                f"{API_URL}/model",
                json={"provider": provider, "model": model},
                timeout=3,
            )
            st.rerun()
    except Exception:
        st.warning("API not available")

    st.divider()
    st.subheader("Stats")
    try:
        stats = requests.get(f"{API_URL}/stats", timeout=3).json()
        col1, col2 = st.columns(2)
        col1.metric("Parent chunks", stats.get("parent_chunks", 0))
        col2.metric("Child chunks", stats.get("child_chunks", 0))
    except Exception:
        st.warning("API not available")

    st.divider()
    st.subheader("Architecture")
    st.markdown(
        "- **Search**: Hybrid (BM25 + Vector + RRF)\n"
        "- **Re-ranking**: LLM-based\n"
        "- **Chunking**: Parent-child\n"
        "- **Embeddings**: Local (all-MiniLM-L6-v2)\n"
        "- **LLM**: Groq / OpenAI (streaming)\n"
        "- **Memory**: Multi-turn with query reformulation"
    )

# Load messages from DB
if st.session_state.chat_id:
    messages = get_messages(st.session_state.chat_id)
else:
    messages = []

for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    src = s.get("source", "unknown")
                    page = s.get("page")
                    label = f"{src}" + (f" (p. {page})" if page is not None else "")
                    st.markdown(f"**{label}**")
                    st.text(s["content"])

if prompt := st.chat_input("Ask a question about your documents"):
    if not st.session_state.chat_id:
        st.session_state.chat_id = create_chat(prompt[:50])

    add_message(st.session_state.chat_id, "user", prompt)

    chats_list = list_chats()
    current = next((c for c in chats_list if c["id"] == st.session_state.chat_id), None)
    if current and current["title"] == "New Chat":
        update_title(st.session_state.chat_id, prompt[:50])

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
        ]

        try:
            resp = requests.post(
                f"{API_URL}/query",
                json={"question": prompt, "history": history, "stream": True},
                stream=True,
                timeout=60,
            )

            answer = ""
            sources = []
            placeholder = st.empty()

            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])

                if data["type"] == "sources":
                    sources = data["sources"]
                elif data["type"] == "token":
                    answer += data["content"]
                    placeholder.markdown(answer + "▌")
                elif data["type"] == "done":
                    placeholder.markdown(answer)

            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        src = s.get("source", "unknown")
                        page = s.get("page")
                        label = f"{src}" + (f" (p. {page})" if page is not None else "")
                        st.markdown(f"**{label}**")
                        st.text(s["content"])

        except Exception as e:
            answer = f"Error: {e}"
            sources = []
            st.error(answer)

    add_message(st.session_state.chat_id, "assistant", answer, sources)
