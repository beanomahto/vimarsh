

import json
import os
import requests
import streamlit as st
import streamlit.components.v1 as components

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chat_history import (
    create_chat, list_chats, get_messages, add_message,
    update_title, delete_chat,
)

# ── Configurable API URL (for deployment) ────────
# API_URL = os.getenv("API_URL", "http://localhost:8000")
API_URL = os.getenv("API_URL", "https://vimarsh-l5x7.onrender.com")

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COLD START LOADING SCREEN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _check_api_health(timeout: int = 5) -> bool:
    """Check if the backend API is alive."""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def _show_loading_screen():
    """Display a premium animated loading screen using components.html()."""

    # Hide all Streamlit default UI elements
    st.markdown("""
    <style>
        #MainMenu, footer, header,
        .stAppHeader, .stAppDeployButton,
        section[data-testid="stSidebar"],
        .stMainBlockContainer > div > div > div:not(:first-child) {
            display: none !important;
        }
        .stMainBlockContainer {
            padding: 0 !important;
            margin: 0 !important;
        }
        .block-container {
            padding: 0 !important;
            max-width: 100% !important;
        }
        .stApp {
            background: #0f0c29 !important;
        }
        iframe {
            border: none !important;
        }
    </style>
    """, unsafe_allow_html=True)

    loading_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            width: 100%;
            height: 100vh;
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            overflow: hidden;
            position: relative;
        }}

        /* ── Animated background ─────────────── */
        body::before {{
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background:
                radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.15) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 50% 50%, rgba(120, 200, 255, 0.08) 0%, transparent 50%);
            animation: bgShift 8s ease-in-out infinite;
            z-index: 0;
        }}

        @keyframes bgShift {{
            0%, 100% {{ transform: translate(0, 0) rotate(0deg); }}
            33% {{ transform: translate(30px, -30px) rotate(1deg); }}
            66% {{ transform: translate(-20px, 20px) rotate(-1deg); }}
        }}

        /* ── Floating particles ──────────────── */
        .particle {{
            position: absolute;
            border-radius: 50%;
            z-index: 1;
            pointer-events: none;
        }}

        .p1 {{ width: 3px; height: 3px; background: rgba(167,139,250,0.5); top: 12%; left: 18%; animation: float1 7s ease-in-out infinite; }}
        .p2 {{ width: 4px; height: 4px; background: rgba(236,72,153,0.35); top: 72%; left: 78%; animation: float2 9s ease-in-out infinite 1s; }}
        .p3 {{ width: 3px; height: 3px; background: rgba(167,139,250,0.4); top: 28%; left: 88%; animation: float3 8s ease-in-out infinite 2s; }}
        .p4 {{ width: 5px; height: 5px; background: rgba(120,200,255,0.3); top: 82%; left: 12%; animation: float1 10s ease-in-out infinite 0.5s; }}
        .p5 {{ width: 2px; height: 2px; background: rgba(167,139,250,0.6); top: 45%; left: 55%; animation: float2 6s ease-in-out infinite 3s; }}
        .p6 {{ width: 3px; height: 3px; background: rgba(236,72,153,0.3); top: 18%; left: 62%; animation: float3 11s ease-in-out infinite 1.5s; }}
        .p7 {{ width: 4px; height: 4px; background: rgba(120,200,255,0.25); top: 65%; left: 30%; animation: float1 8.5s ease-in-out infinite 2.5s; }}

        @keyframes float1 {{
            0%, 100% {{ transform: translateY(0) translateX(0); opacity: 0.4; }}
            25% {{ transform: translateY(-35px) translateX(20px); opacity: 0.9; }}
            50% {{ transform: translateY(-15px) translateX(-25px); opacity: 0.3; }}
            75% {{ transform: translateY(-45px) translateX(15px); opacity: 0.7; }}
        }}
        @keyframes float2 {{
            0%, 100% {{ transform: translateY(0) translateX(0); opacity: 0.3; }}
            30% {{ transform: translateY(25px) translateX(-15px); opacity: 0.8; }}
            60% {{ transform: translateY(-20px) translateX(30px); opacity: 0.4; }}
        }}
        @keyframes float3 {{
            0%, 100% {{ transform: translateY(0) translateX(0); opacity: 0.5; }}
            40% {{ transform: translateY(-30px) translateX(-20px); opacity: 0.7; }}
            70% {{ transform: translateY(15px) translateX(25px); opacity: 0.3; }}
        }}

        /* ── Glowing orb ─────────────────────── */
        .orb-container {{
            position: relative;
            width: 120px;
            height: 120px;
            margin-bottom: 48px;
            z-index: 2;
        }}

        .orb {{
            width: 120px;
            height: 120px;
            border-radius: 50%;
            background: radial-gradient(circle at 35% 35%,
                rgba(196, 181, 253, 0.95),
                rgba(139, 92, 246, 0.75),
                rgba(109, 40, 217, 0.55));
            box-shadow:
                0 0 60px rgba(139, 92, 246, 0.5),
                0 0 120px rgba(139, 92, 246, 0.25),
                inset 0 0 40px rgba(255, 255, 255, 0.12);
            animation: orbPulse 2.5s ease-in-out infinite;
        }}

        @keyframes orbPulse {{
            0%, 100% {{
                transform: scale(1);
                box-shadow:
                    0 0 60px rgba(139, 92, 246, 0.5),
                    0 0 120px rgba(139, 92, 246, 0.25);
            }}
            50% {{
                transform: scale(1.1);
                box-shadow:
                    0 0 90px rgba(139, 92, 246, 0.65),
                    0 0 180px rgba(139, 92, 246, 0.35);
            }}
        }}

        /* ── Orbit rings ─────────────────────── */
        .ring1 {{
            position: absolute;
            top: -18px; left: -18px;
            width: 156px; height: 156px;
            border: 2px solid transparent;
            border-top-color: rgba(167, 139, 250, 0.6);
            border-right-color: rgba(167, 139, 250, 0.25);
            border-radius: 50%;
            animation: spin 3s linear infinite;
        }}

        .ring2 {{
            position: absolute;
            top: -30px; left: -30px;
            width: 180px; height: 180px;
            border: 1.5px solid transparent;
            border-bottom-color: rgba(236, 72, 153, 0.45);
            border-left-color: rgba(236, 72, 153, 0.2);
            border-radius: 50%;
            animation: spin 5s linear infinite reverse;
        }}

        .ring3 {{
            position: absolute;
            top: -42px; left: -42px;
            width: 204px; height: 204px;
            border: 1px solid transparent;
            border-top-color: rgba(120, 200, 255, 0.25);
            border-radius: 50%;
            animation: spin 8s linear infinite;
        }}

        @keyframes spin {{
            from {{ transform: rotate(0deg); }}
            to {{ transform: rotate(360deg); }}
        }}

        /* ── Orbit dots ──────────────────────── */
        .dot {{
            position: absolute;
            width: 7px; height: 7px;
            border-radius: 50%;
            box-shadow: 0 0 8px currentColor;
        }}
        .dot1 {{
            color: rgba(167, 139, 250, 0.9);
            background: currentColor;
            top: -6px; left: 50%;
            animation: dotPulse 3s linear infinite;
        }}
        .dot2 {{
            color: rgba(236, 72, 153, 0.8);
            background: currentColor;
            bottom: -4px; right: 15%;
            animation: dotPulse 3s linear infinite 1.5s;
        }}
        .dot3 {{
            color: rgba(120, 200, 255, 0.7);
            background: currentColor;
            top: 40%; left: -8px;
            width: 5px; height: 5px;
            animation: dotPulse 3s linear infinite 0.8s;
        }}

        @keyframes dotPulse {{
            0%, 100% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.3; transform: scale(0.5); }}
        }}

        /* ── Text ────────────────────────────── */
        .content {{
            text-align: center;
            z-index: 2;
        }}

        .title {{
            font-size: 30px;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 10px;
            letter-spacing: -0.5px;
            animation: fadeUp 0.8s ease-out;
        }}

        .subtitle {{
            font-size: 15px;
            font-weight: 400;
            color: rgba(255, 255, 255, 0.55);
            margin-bottom: 40px;
            animation: fadeUp 0.8s ease-out 0.15s both;
            line-height: 1.5;
        }}

        @keyframes fadeUp {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        /* ── Progress bar ────────────────────── */
        .progress-track {{
            width: 300px;
            height: 4px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 24px;
            z-index: 2;
            animation: fadeUp 0.8s ease-out 0.3s both;
        }}

        .progress-fill {{
            height: 100%;
            width: 35%;
            background: linear-gradient(90deg,
                rgba(139, 92, 246, 0.9),
                rgba(236, 72, 153, 0.9),
                rgba(120, 200, 255, 0.9),
                rgba(139, 92, 246, 0.9));
            background-size: 300% 100%;
            border-radius: 4px;
            animation: progressSlide 2.2s ease-in-out infinite;
        }}

        @keyframes progressSlide {{
            0% {{ transform: translateX(-120%); background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ transform: translateX(400%); background-position: 0% 50%; }}
        }}

        /* ── Status footer ───────────────────── */
        .status {{
            font-size: 13px;
            color: rgba(255, 255, 255, 0.35);
            z-index: 2;
            animation: fadeUp 0.8s ease-out 0.45s both;
            text-align: center;
        }}

        .status-line2 {{
            margin-top: 6px;
            font-size: 12px;
            color: rgba(255, 255, 255, 0.25);
        }}

        #attempt-counter {{
            color: rgba(167, 139, 250, 0.6);
            font-weight: 500;
        }}

        .dots span {{
            animation: blink 1.4s infinite both;
            font-size: 20px;
            line-height: 1;
            vertical-align: middle;
        }}
        .dots span:nth-child(2) {{ animation-delay: 0.2s; }}
        .dots span:nth-child(3) {{ animation-delay: 0.4s; }}

        @keyframes blink {{
            0% {{ opacity: 0.2; }}
            20% {{ opacity: 1; }}
            100% {{ opacity: 0.2; }}
        }}

        /* ── Error state ─────────────────────── */
        .error-msg {{
            display: none;
            margin-top: 20px;
            padding: 14px 24px;
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 10px;
            color: rgba(255, 255, 255, 0.8);
            font-size: 14px;
            z-index: 2;
            animation: fadeUp 0.5s ease-out;
        }}
        .error-msg a {{
            color: rgba(167, 139, 250, 0.9);
            text-decoration: none;
            font-weight: 500;
        }}
        .error-msg a:hover {{ text-decoration: underline; }}
    </style>
    </head>

    <body>
        <!-- Particles -->
        <div class="particle p1"></div>
        <div class="particle p2"></div>
        <div class="particle p3"></div>
        <div class="particle p4"></div>
        <div class="particle p5"></div>
        <div class="particle p6"></div>
        <div class="particle p7"></div>

        <!-- Orb -->
        <div class="orb-container">
            <div class="orb"></div>
            <div class="ring1"></div>
            <div class="ring2"></div>
            <div class="ring3"></div>
            <div class="dot dot1"></div>
            <div class="dot dot2"></div>
            <div class="dot dot3"></div>
        </div>

        <!-- Text -->
        <div class="content">
            <div class="title">Waking Up the Server</div>
            <div class="subtitle">
                Our free-tier server is spinning up — hang tight!
            </div>
        </div>

        <!-- Progress -->
        <div class="progress-track">
            <div class="progress-fill"></div>
        </div>

        <!-- Status -->
        <div class="status">
            <span id="status-text">This usually takes 30–60 seconds on cold start</span>
            <span class="dots"><span>.</span><span>.</span><span>.</span></span>
            <div class="status-line2">
                <span id="attempt-counter">Connecting...</span>
            </div>
        </div>

        <!-- Error (hidden until timeout) -->
        <div class="error-msg" id="error-box">
            Server didn't respond after multiple attempts.<br>
            <a href="javascript:window.parent.location.reload()">Click here to retry</a>
        </div>

        <script>
            const API = "{API_URL}";
            let attempt = 0;
            const maxAttempts = 40;

            function check() {{
                attempt++;
                const counter = document.getElementById('attempt-counter');
                counter.textContent = 'Attempt ' + attempt + ' of ' + maxAttempts;

                fetch(API + '/health', {{ mode: 'cors' }})
                    .then(function(r) {{
                        if (r.ok) {{
                            document.getElementById('status-text').textContent = 'Server is ready! Loading app';
                            counter.textContent = 'Connected ✓';
                            // Small delay so user sees "Connected" message
                            setTimeout(function() {{
                                window.parent.location.reload();
                            }}, 800);
                        }} else {{
                            scheduleNext();
                        }}
                    }})
                    .catch(function() {{
                        scheduleNext();
                    }});
            }}

            function scheduleNext() {{
                if (attempt < maxAttempts) {{
                    setTimeout(check, 3000);
                }} else {{
                    document.getElementById('status-text').textContent = 'Server is taking longer than expected';
                    document.getElementById('attempt-counter').textContent = 'Timed out';
                    document.getElementById('error-box').style.display = 'block';
                }}
            }}

            // Start polling after 2 seconds
            setTimeout(check, 2000);
        </script>
    </body>
    </html>
    """

    components.html(loading_html, height=700, scrolling=False)
    st.stop()  # Don't render anything else


# ── Check API health before rendering the app ───
if not _check_api_health(timeout=5):
    _show_loading_screen()
    # st.stop() is called inside _show_loading_screen()
    # so nothing below this runs until the API is up


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN APP (only runs after API is confirmed alive)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.title("Vimarsh: AI Assistant")
st.caption("Retrieval-augmented generation (RAG) + hybrid search pgvector + LLM re-ranking")

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
                    st.success(
                        f"{uploaded_file.name}: {data['chunks']} chunks indexed"
                    )
                else:
                    st.error(f"Failed to process {uploaded_file.name}")

    st.divider()
    st.subheader("Model")
    try:
        prov_data = requests.get(f"{API_URL}/providers", timeout=3).json()
        providers = prov_data["providers"]
        current = prov_data["current"]

        available = [p for p, cfg in providers.items() if cfg["available"]]
        st.selectbox(
            "Provider",
            ["llama3:7b"],
            index=0,
            key="provider_select",
        )
        provider="openai"

        models = providers[provider]["models"]
        cur_model = (
            current["model"]
            if current["provider"] == provider
            else providers[provider]["default_model"]
        )
        st.selectbox(
            "Provider",
            ["ollama3:3b", "ollama3:7b", "ollama3:27b"],
            index=0,
            key="provider_select",
        )
        provider="openai"
        model="gpt-4o-mini"

        if provider != current["provider"] or model != current["model"]:
            requests.post(
                f"{API_URL}/model",
                json={"provider": provider, "model": model},
                timeout=3,
            )
            st.rerun()
    except Exception:
        st.warning("ollama hallucinations injected")

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
    st.subheader("RAG Architecture")
    st.markdown(
        "- **Vector**: pgvector (Supabase)\n"
        "- **Keyword**: PostgreSQL full-text search\n"
        "- **Fusion**: RRF + LLM re-ranking\n"
        "- **Chunking**: Parent-child\n"
        "- **Embeddings**: `text-embedding-3-small`\n"
        "- **LLM**: Groq (streaming)\n"
        "- **Memory**: Multi-turn with query reformulation\n"
        "- **Created By**: Intern-Bhawani Mahato"
    )

# Load messages from DB
if st.session_state.chat_id:
    messages = get_messages(st.session_state.chat_id)
else:
    messages = []

# Display existing messages
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

    # ── 1. Show user message IMMEDIATELY ─────────
    with st.chat_message("user"):
        st.markdown(prompt)

    # ── 2. Show assistant "thinking" right away ──
    with st.chat_message("assistant"):
        # Show spinner WHILE doing DB + API work
        with st.spinner("Thinking..."):

            # Create chat if needed
            if not st.session_state.chat_id:
                st.session_state.chat_id = create_chat(prompt[:50])

            # Save user message to DB
            add_message(st.session_state.chat_id, "user", prompt)

            # Auto-title if "New Chat"
            chats_list = list_chats()
            current = next(
                (c for c in chats_list if c["id"] == st.session_state.chat_id), None
            )
            if current and current["title"] == "New Chat":
                update_title(st.session_state.chat_id, prompt[:50])

            # Build history for context
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in messages
            ]

            # Call the API
            try:
                resp = requests.post(
                    f"{API_URL}/query",
                    json={"question": prompt, "history": history, "stream": True},
                    stream=True,
                    timeout=120,
                )
            except Exception as e:
                st.error(f"API connection failed: {e}")
                add_message(st.session_state.chat_id, "assistant", f"Error: {e}")
                st.stop()

        # ── 3. Stream the response with live tokens ──
        answer = ""
        sources = []
        placeholder = st.empty()

        try:
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])

                if data["type"] == "sources":
                    sources = data["sources"]
                elif data["type"] == "token":
                    answer += data["content"]
                    placeholder.markdown(answer + " ▌")
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

        # ── 4. Save assistant response to DB ─────
        add_message(st.session_state.chat_id, "assistant", answer, sources)
