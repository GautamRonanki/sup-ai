# SupAI

SupAI is a lightweight RAG (Retrieval-Augmented Generation) application that lets you query any document or article using natural language — without reading the whole thing.

Paste a link or upload a document, ask a question, and SupAI pulls the most relevant information and answers it for you.

---

## What it does

If you're looking for something specific within a long document or article, SupAI saves you from reading everything. You give it a source, ask questions, and it finds the answers from within that source — nothing else.

---

## How it works

### 1. Ingestion
When you provide a link or upload a document, SupAI scrapes or reads the content and breaks it into small overlapping chunks. Each chunk is converted into a vector (embedding) using OpenAI's `text-embedding-3-small` model and stored in memory for the session.

### 2. Query
When you ask a question, SupAI:
- **Rewrites your question** using an LLM to fix spelling mistakes and improve clarity before retrieval
- **Converts your question** into a vector using the same embedding model
- **Finds the most relevant chunks** by calculating cosine similarity between your question vector and all stored chunk vectors
- **Retrieves the top 3 chunks** as context

### 3. Generation
The retrieved chunks are sent to `gpt-4o-mini` along with your question. The LLM answers using only the provided context and cites its sources explicitly.

### 4. Classification & Error Handling
Every query is classified at two stages:
- **Retrieval classification** — checks if the similarity scores are high enough to be useful. If chunks are too dissimilar from the question, retrieval is marked as failed and the LLM is not called for generation.
- **Generation classification** — uses LLM-as-a-judge to determine if the generated answer was confident, hedged, or a refusal.

If retrieval fails, SupAI tells you what the source actually contains and suggests a more relevant question instead of guessing.

---

## Key Design Decisions

**No pre-loaded knowledge base.** SupAI doesn't ship with any documents. You bring your own source every session. This keeps it flexible and avoids stale or irrelevant data.

**Query rewriting before retrieval.** User questions are often imprecise or have typos. Running the question through an LLM first improves retrieval quality significantly — especially for proper nouns and domain-specific terms.

**LLM-as-a-judge for generation classification.** Instead of checking if the answer contains phrases like "I don't know", an LLM reads the full answer and classifies it. This is more robust and catches refusals regardless of how they are phrased.

**Fail fast on poor retrieval.** If similarity scores are too low, SupAI skips generation entirely and redirects the user. This avoids wasting tokens on a generation that will likely hallucinate or refuse anyway.

**Session-based storage.** The index lives in memory for the duration of the session. No data is persisted between sessions, which keeps the system simple and avoids stale embeddings.

---

## Known Limitations

**Dense documents with repeated entities hurt retrieval.** If a document mentions the same topic many times across different contexts (e.g. a Wikipedia article), the retrieved chunks may be topically related but not contain the specific fact being asked about. For example, asking "What is the capital of Maharashtra?" on a document about Indian states may retrieve chunks about Maharashtra's history or politics rather than its capital city.

**Three LLM calls per query.** Every query triggers query rewriting, answer generation, and generation classification — three separate API calls. This adds latency (~0.8-1s per query) and cost compared to a simpler pipeline.

**No persistent index.** Embeddings are rebuilt every session. For large documents, this means a noticeable wait before you can start querying.

**Similarity thresholds are fixed.** The cutoffs for confident (≥0.7), uncertain (0.4-0.7), and failed (<0.4) retrieval are hardcoded. These may not be appropriate for all document types or query styles.

---

## Future Improvements

- **Re-ranking with LLM** — after retrieval, ask an LLM to verify whether the retrieved chunks actually contain the answer before generating. This would catch cases where similarity scores look reasonable but the chunks are not truly relevant.
- **Persistent index** — cache embeddings to disk so the same document doesn't need to be re-embedded every session.
- **Adjustable retrieval thresholds** — let users or the system tune similarity cutoffs based on document type.

---

## Evaluation

SupAI was evaluated against a test set of 9 questions on a Wikipedia article about Indian states and union territories.

| Metric | Value |
|---|---|
| Average score | 3.78 / 5 |
| Average retrieval latency | 0.185s |
| Average generation latency | 0.861s |
| Total cost for 9 queries | $0.001064 |

Generation is the primary latency driver (~5x slower than retrieval). Queries where retrieval fails respond faster because the simpler `handle_refusal` path is used instead of full generation.

---

## Tech Stack

- Python + Streamlit
- OpenAI `gpt-4o-mini` for generation, classification, and query rewriting
- OpenAI `text-embedding-3-small` for embeddings
- Cosine similarity for retrieval (no vector database)
- JSON for session-based index storage