# SupAI

A session-based RAG (Retrieval-Augmented Generation) tool built with Streamlit. Paste a link or upload documents, and ask questions about the content using AI.

## What it does

1. **Add content** — Paste a URL or upload files (PDF, TXT, DOC, DOCX, CSV) from the sidebar
2. **Processing** — The app extracts text, splits it into chunks, and creates vector embeddings
3. **Chat** — Ask questions and get answers grounded in your uploaded content, with source citations

All data is session-based — nothing is stored on disk. Closing the browser tab clears everything.

## Prerequisites

- **Python 3.10+** — [Download here](https://www.python.org/downloads/)
- **OpenAI API key** — [Get one here](https://platform.openai.com/api-keys)

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd SupAI
```

### 2. Create a virtual environment

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

You should see `(.venv)` in your terminal prompt after activation.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your OpenAI API key

Create a `.env` file in the project root:

```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

Replace `your-api-key-here` with your actual OpenAI API key. The file should look like:

```
OPENAI_API_KEY=sk-proj-...
```

> **Important:** Never commit your `.env` file to git. Add it to `.gitignore`:
> ```bash
> echo ".env" >> .gitignore
> ```

### 5. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## How to use

1. **Paste a link** — Enter a URL in the sidebar and click "Fetch & Analyze". The app will scrape the article text from the page.

2. **Upload documents** — Use the file uploader in the sidebar to upload PDF, TXT, DOC, DOCX, or CSV files. Click "Upload & Analyze" to process them.

3. **Ask questions** — Once your content is processed, type a question in the chat input at the bottom. The AI will answer using only the content you provided, with source citations.

4. **Add more content** — You can add more links or files at any time from the sidebar. New content is added to your existing session.

5. **Start fresh** — Click "New Session" in the sidebar to clear everything and start over.

## Session limits

- **Budget:** $0.10 per session (covers OpenAI API costs for embeddings + chat)
- **Sources:** Up to 5 sources per session
- **File size:** Max 10 MB per file
- **Chunks:** Max 500 text chunks in the index

## Project structure

```
SupAI/
├── app.py                 # Main Streamlit app (UI + state management)
├── upload_utils.py        # URL scraping, file parsing, embedding creation
├── chunk_articles.py      # Text chunking logic
├── retrieval.py           # Vector similarity search
├── rag_pipeline.py        # LLM answer generation (GPT-4o-mini)
├── requirements.txt       # Python dependencies
├── .env                   # Your OpenAI API key (not committed)
└── README.md              # This file
```

## Tech stack

- **Frontend:** Streamlit
- **LLM:** OpenAI GPT-4o-mini
- **Embeddings:** OpenAI text-embedding-3-small
- **Web scraping:** Trafilatura
- **PDF parsing:** PyPDF2
- **Word docs:** python-docx
- **Vector search:** NumPy cosine similarity (in-memory)

## Troubleshooting

**"No module named ..."**
Make sure your virtual environment is activated (`source .venv/bin/activate`) and you've run `pip install -r requirements.txt`.

**"Could not fetch URL"**
Some websites block automated scraping (LinkedIn, sites behind logins, etc.). Try a public article or blog post instead.

**"Session budget exceeded"**
The $0.10 per session limit has been reached. Click "New Session" in the sidebar to start fresh.

**Port already in use**
If `localhost:8501` is taken, Streamlit will automatically try the next port (8502, 8503, etc.). Check your terminal output for the actual URL.
