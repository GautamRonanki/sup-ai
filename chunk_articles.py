def chunk_article(text, filename):
    """Split article text into paragraph chunks."""

    # Replace single newlines with spaces (joins broken lines)
    # But keep actual paragraph breaks
    lines = text.split("\n")

    chunks = []
    current_chunk = ""

    for line in lines:
        line = line.strip()

        # Skip empty lines or obvious metadata
        if not line or len(line) < 10:
            continue
        if "min. read" in line.lower() or "view original" in line.lower():
            continue

        # Add line to current chunk
        current_chunk += " " + line

        # Check if this looks like end of paragraph
        # (ends with punctuation and is long enough)
        if line.endswith((".", "!", "?", '"', "'")):
            if len(current_chunk.strip()) > 100:
                chunks.append(
                    {
                        "text": current_chunk.strip(),
                        "source": filename,
                        "chunk_id": len(chunks),
                    }
                )
                current_chunk = ""

    # Add last chunk if it exists
    if len(current_chunk.strip()) > 100:
        chunks.append(
            {"text": current_chunk.strip(), "source": filename, "chunk_id": len(chunks)}
        )

    return chunks


def chunk_all_articles(articles):
    """Chunk all articles into paragraphs."""
    all_chunks = []

    for article in articles:
        chunks = chunk_article(article["text"], article["filename"])
        all_chunks.extend(chunks)
        print(f"✓ {article['filename']}: {len(chunks)} chunks")

    return all_chunks
