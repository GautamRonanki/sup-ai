import io
import streamlit as st
import trafilatura
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
from chunk_articles import chunk_article

load_dotenv()
client = OpenAI()

# Pricing per token
EMBED_COST_PER_TOKEN = 0.02 / 1_000_000  # text-embedding-3-small


def scrape_url(url):
    """Fetch and extract article text from a URL."""
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        raise ValueError(f"Could not fetch URL: {url}")

    text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
    if not text or len(text.strip()) < 50:
        raise ValueError("Could not extract meaningful text from this URL")

    return {"text": text, "source": url, "filename": url}


def process_file_bytes(name, data):
    """Extract text from uploaded file bytes (PDF, TXT, CSV, DOC, DOCX)."""
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""

    if ext == "pdf":
        reader = PdfReader(io.BytesIO(data))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

    elif ext == "txt":
        text = data.decode("utf-8", errors="ignore")

    elif ext == "csv":
        text = data.decode("utf-8", errors="ignore")

    elif ext in ("doc", "docx"):
        import docx
        doc = docx.Document(io.BytesIO(data))
        text = "\n".join(p.text for p in doc.paragraphs)

    else:
        raise ValueError(f"Unsupported file type: .{ext}")

    if not text.strip():
        raise ValueError(f"No text could be extracted from {name}")

    return {"filename": name, "text": text}


def chunk_uploaded_articles(articles):
    """Chunk all articles into paragraphs."""
    all_chunks = []
    for article in articles:
        chunks = chunk_article(article["text"], article["filename"])
        all_chunks.extend(chunks)
    return all_chunks


def create_embeddings_with_progress(chunks, batch_size=100):
    """Create embeddings for all chunks in batches with progress indicator.

    Returns:
        tuple: (embeddings_list, total_cost)
    """
    all_embeddings = []
    total_cost = 0.0
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    st.write(f"Creating embeddings for {len(chunks)} chunks...")
    progress_bar = st.progress(0)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [chunk["text"] for chunk in batch]

        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )

        embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(embeddings)
        total_cost += response.usage.total_tokens * EMBED_COST_PER_TOKEN

        current_batch = (i // batch_size) + 1
        progress_bar.progress(current_batch / total_batches)

    return all_embeddings, total_cost
