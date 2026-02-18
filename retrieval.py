import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# Pricing per token
EMBED_COST_PER_TOKEN = 0.02 / 1_000_000  # text-embedding-3-small


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def retrieve_relevant_chunks(question, index_data, top_k=5):
    """Find the most relevant chunks for a given question.

    Returns:
        tuple: (top_chunks, cost) where cost is the embedding API cost
    """
    # Convert question to embedding
    response = client.embeddings.create(input=question, model="text-embedding-3-small")
    question_embedding = response.data[0].embedding
    cost = response.usage.total_tokens * EMBED_COST_PER_TOKEN

    # Calculate similarity with all chunks
    similarities = []
    for item in index_data:
        similarity = cosine_similarity(question_embedding, item["embedding"])
        similarities.append(
            {"text": item["text"], "source": item["source"], "similarity": similarity}
        )

    # Sort by similarity and get top K
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    top_chunks = similarities[:top_k]

    return top_chunks, cost
