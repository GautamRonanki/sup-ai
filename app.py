import streamlit as st
import streamlit.components.v1 as components
from upload_utils import (
    scrape_url,
    process_file_bytes,
    create_embeddings_with_progress,
)
from chunk_articles import chunk_article
from retrieval import retrieve_relevant_chunks
from rag_pipeline import generate_answer

# ==========================================
# LIMITS
# ==========================================
SESSION_BUDGET = 0.10        # Max $0.10 per session
MAX_SOURCES = 5              # Max 5 sources per session
MAX_FILE_SIZE_MB = 10        # Max 10MB per file
MAX_CHUNKS = 500             # Max 500 chunks in index

# Page config
st.set_page_config(
    page_title="SupAI",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto",
)

# ==========================================
# GLOBAL STYLES
# ==========================================
st.markdown(
    """
    <style>
    /* Hide hamburger & deploy menu */
    #MainMenu { display: none !important; }

    /* Tighten top padding */
    .block-container { padding-top: 2rem !important; padding-bottom: 1rem !important; }

    /* ---- SIDEBAR ---- */
    [data-testid="stSidebar"] {
        background: #fafafa;
        border-right: 1px solid #eee;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem !important;
    }
    .sidebar-brand {
        font-size: 1.4rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0.25rem;
    }
    .sidebar-tagline {
        font-size: 0.75rem;
        color: #999;
        margin-bottom: 1.5rem;
    }
    .sidebar-section-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #555;
        margin-bottom: 0.25rem;
    }
    .sidebar-divider {
        border: none;
        border-top: 1px solid #eee;
        margin: 1rem 0;
    }

    /* Source chips in sidebar */
    .source-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        padding: 0.25rem 0;
    }
    .source-chip {
        background: #fff;
        border: 1px solid #e0e3e9;
        border-radius: 999px;
        padding: 0.2rem 0.6rem;
        font-size: 0.68rem;
        color: #555;
        max-width: 220px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    /* ---- ENTRY PAGE ---- */
    .brand { text-align: center; margin-top: 20vh; margin-bottom: 2rem; }
    .brand h1 { font-size: 3rem; font-weight: 700; margin: 0; letter-spacing: -1px; }
    .brand p { color: #999; font-size: 1rem; margin-top: 0.25rem; }

    /* ---- Button tweaks ---- */
    .stButton > button[kind="primary"] {
        border-radius: 8px;
        font-weight: 500;
        padding: 0.5rem 1.5rem;
    }

    /* ---- CHAT PAGE ---- */

    /* Chat messages */
    [data-testid="stChatMessage"] {
        padding: 0.85rem 1rem;
        border-radius: 10px;
        margin-bottom: 0.25rem;
    }

    /* User message background */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: #f7f7f8;
    }

    /* Assistant message */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: transparent;
    }

    /* Source expander inside chat */
    [data-testid="stChatMessage"] [data-testid="stExpander"] {
        border: none !important;
        background: transparent !important;
    }
    [data-testid="stChatMessage"] [data-testid="stExpander"] summary {
        font-size: 0.78rem;
        color: #888;
    }

    /* Welcome card */
    .welcome-card {
        text-align: center;
        padding: 4rem 2rem 2rem 2rem;
        color: #aaa;
    }
    .welcome-card h4 {
        color: #777;
        font-weight: 500;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    .welcome-card p { font-size: 0.85rem; margin: 0.2rem 0; }

    /* ---- MOBILE ---- */
    @media (max-width: 768px) {
        .brand { margin-top: 6vh !important; }
        .brand h1 { font-size: 2.2rem !important; }
        .brand p { font-size: 0.88rem !important; }
        .welcome-card { padding: 1.5rem 0.5rem 1rem 0.5rem !important; }
        .welcome-card h4 { font-size: 1rem !important; }
        .block-container {
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }
        [data-testid="stChatMessage"] { padding: 0.6rem 0.75rem !important; }
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# SESSION STATE
# ==========================================
if "app_state" not in st.session_state:
    st.session_state.app_state = "entry"
    st.session_state.index_data = []
    st.session_state.sources = set()
    st.session_state.messages = []
    st.session_state.processing_input = None
    st.session_state.session_cost = 0.0


def budget_exceeded():
    return st.session_state.session_cost >= SESSION_BUDGET


# ==========================================
# SIDEBAR (always visible)
# ==========================================
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🤖 SupAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Paste a link or upload docs, then ask anything.</div>', unsafe_allow_html=True)

    # ---- Link input ----
    if st.session_state.pop("_clear_url", False) and "sidebar_url" in st.session_state:
        del st.session_state["sidebar_url"]
    st.markdown('<div class="sidebar-section-label">🔗 Paste a link</div>', unsafe_allow_html=True)
    url_input = st.text_input(
        "url", placeholder="https://example.com/article",
        label_visibility="collapsed", key="sidebar_url",
    )
    url_go = st.button("Fetch & Analyze", type="primary", use_container_width=True, key="sidebar_url_btn")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # ---- File upload ----
    st.markdown('<div class="sidebar-section-label">📄 Upload documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "files", type=["pdf", "txt", "doc", "docx", "csv"],
        accept_multiple_files=True,
        label_visibility="collapsed", key="sidebar_files",
    )
    file_go = st.button(
        "Upload & Analyze", type="primary", use_container_width=True,
        disabled=not uploaded_files, key="sidebar_files_btn",
    )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # ---- Loaded sources ----
    src_list = sorted(st.session_state.sources)
    if src_list:
        st.markdown('<div class="sidebar-section-label">📚 Sources</div>', unsafe_allow_html=True)
        chips_html = ""
        for s in src_list:
            label = s if len(s) < 40 else s[:37] + "..."
            chips_html += f'<span class="source-chip">{label}</span>'
        st.markdown(f'<div class="source-chips">{chips_html}</div>', unsafe_allow_html=True)
        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # ---- New session button ----
    if st.button("✕ New Session", use_container_width=True, key="sidebar_new"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ==========================================
# SIDEBAR ACTIONS (handle submissions)
# ==========================================
sidebar_error = None

if url_go and url_input:
    url_stripped = url_input.strip()
    if not (url_stripped.startswith("http://") or url_stripped.startswith("https://")):
        sidebar_error = "Please enter a valid URL starting with http:// or https://"
    elif budget_exceeded():
        sidebar_error = "Session budget exceeded. Please start a new session."
    elif len(st.session_state.sources) >= MAX_SOURCES:
        sidebar_error = f"Maximum {MAX_SOURCES} sources per session."
    else:
        st.session_state.processing_input = {"type": "url", "value": url_stripped}
        st.session_state.app_state = "processing"
        st.session_state["_clear_url"] = True
        st.rerun()

if file_go and uploaded_files:
    oversized = [f.name for f in uploaded_files if f.size > MAX_FILE_SIZE_MB * 1024 * 1024]
    if oversized:
        sidebar_error = f"Files too large (max {MAX_FILE_SIZE_MB}MB): {', '.join(oversized)}"
    elif budget_exceeded():
        sidebar_error = "Session budget exceeded. Please start a new session."
    elif len(st.session_state.sources) + len(uploaded_files) > MAX_SOURCES:
        sidebar_error = f"Maximum {MAX_SOURCES} sources per session. You have {len(st.session_state.sources)} already."
    else:
        file_data = [{"name": f.name, "bytes": f.read()} for f in uploaded_files]
        st.session_state.processing_input = {"type": "files", "value": file_data}
        st.session_state.app_state = "processing"
        del st.session_state["sidebar_files"]
        st.rerun()

if sidebar_error:
    with st.sidebar:
        st.error(sidebar_error)


# ==========================================
# ENTRY STATE
# ==========================================
if st.session_state.app_state == "entry":

    st.markdown(
        '<div class="brand">'
        '<h1>🤖 SupAI</h1>'
        '<p>Paste a link or upload documents in the sidebar — then ask anything.</p>'
        '</div>',
        unsafe_allow_html=True,
    )


# ==========================================
# PROCESSING STATE
# ==========================================
elif st.session_state.app_state == "processing":
    pending = st.session_state.processing_input

    if pending is None:
        st.session_state.app_state = "chat" if st.session_state.index_data else "entry"
        st.rerun()

    # Close sidebar on mobile when processing starts
    components.html(
        """
        <script>
        (function() {
            var btn = window.parent.document.querySelector(
                '[data-testid="stSidebarCollapseButton"] button'
            );
            if (btn && window.parent.innerWidth <= 768) btn.click();
        })();
        </script>
        """,
        height=0,
    )

    status = st.status("Processing content...", expanded=True)
    with status:
        articles = []
        failed = False

        if pending["type"] == "url":
            st.write("🌐 Fetching article...")
            try:
                article = scrape_url(pending["value"])
                articles.append(article)
                st.write("✓ Extracted text")
            except Exception as e:
                st.error(str(e))
                failed = True

        elif pending["type"] == "files":
            st.write("📄 Reading PDFs...")
            progress = st.progress(0)
            for i, item in enumerate(pending["value"]):
                try:
                    article = process_file_bytes(item["name"], item["bytes"])
                    articles.append(article)
                    st.write(f"✓ {item['name']}")
                except Exception as e:
                    st.error(f"✗ {item['name']}: {e}")
                progress.progress((i + 1) / len(pending["value"]))

        if failed or not articles:
            status.update(label="Failed", state="error")

    # Error handling outside status block
    if failed or not articles:
        st.session_state.processing_input = None
        st.error("Could not process the content. Please try a different link or file.")
        if st.button("← Go Back"):
            st.session_state.app_state = "chat" if st.session_state.index_data else "entry"
            st.rerun()
        st.stop()

    with status:
        st.write("✂️ Chunking text...")
        all_chunks = []
        for article in articles:
            chunks = chunk_article(article["text"], article["filename"])
            all_chunks.extend(chunks)
        st.write(f"→ {len(all_chunks)} chunks")

        if not all_chunks:
            status.update(label="No chunks", state="error")

    if not all_chunks:
        st.session_state.processing_input = None
        st.error("Could not extract enough text to create chunks.")
        if st.button("← Go Back"):
            st.session_state.app_state = "chat" if st.session_state.index_data else "entry"
            st.rerun()
        st.stop()

    # Check chunk limit
    current_chunks = len(st.session_state.index_data)
    if current_chunks + len(all_chunks) > MAX_CHUNKS:
        st.warning(f"Chunk limit reached. Trimming to {MAX_CHUNKS} total chunks.")
        all_chunks = all_chunks[:MAX_CHUNKS - current_chunks]

    # Embedding + index building
    with status:
        embeddings, embed_cost = create_embeddings_with_progress(all_chunks)
        st.session_state.session_cost += embed_cost
        st.write(f"Embedding cost: ${embed_cost:.4f}")

        st.write("📦 Building index...")
        for chunk, emb in zip(all_chunks, embeddings):
            st.session_state.index_data.append({
                "text": chunk["text"], "source": chunk["source"],
                "chunk_id": chunk["chunk_id"], "embedding": emb,
            })
            st.session_state.sources.add(chunk["source"])

    status.update(label="Done!", state="complete")

    st.session_state.app_state = "chat"
    st.session_state.processing_input = None
    st.rerun()


# ==========================================
# CHAT STATE
# ==========================================
elif st.session_state.app_state == "chat":

    # ---- Welcome card (empty chat) ----
    if not st.session_state.messages:
        st.markdown(
            '<div class="welcome-card">'
            "<h4>Ask anything about your content</h4>"
            "<p>Your sources are loaded and ready to go.</p>"
            "<p>Try asking a specific question about the material.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ---- Message history ----
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📎 Sources"):
                    for i, src in enumerate(msg["sources"], 1):
                        src_label = src["source"] if len(src["source"]) < 50 else src["source"][:47] + "..."
                        st.markdown(f"**{i}.** {src_label} · `{src['similarity']:.3f}`")
                        st.caption(src["text"][:200] + "...")

    # ---- Chat input ----
    if budget_exceeded():
        st.warning("Session budget exceeded ($0.10). Start a new session to continue.")
    else:
        user_input = st.chat_input("Ask a question about your content...")

        if user_input:
            # User message — immediately visible
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Assistant message
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    chunks, retrieval_cost = retrieve_relevant_chunks(
                        user_input, st.session_state.index_data, top_k=3
                    )
                    answer, llm_cost = generate_answer(user_input, chunks)
                    st.session_state.session_cost += retrieval_cost + llm_cost

                st.markdown(answer)
                with st.expander("📎 Sources"):
                    for i, src in enumerate(chunks, 1):
                        src_label = src["source"] if len(src["source"]) < 50 else src["source"][:47] + "..."
                        st.markdown(f"**{i}.** {src_label} · `{src['similarity']:.3f}`")
                        st.caption(src["text"][:200] + "...")

            st.session_state.messages.append({
                "role": "assistant", "content": answer, "sources": chunks,
            })
