from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# Pricing per token
LLM_INPUT_COST_PER_TOKEN = 0.15 / 1_000_000   # gpt-4o-mini input
LLM_OUTPUT_COST_PER_TOKEN = 0.60 / 1_000_000   # gpt-4o-mini output


def generate_answer(question, retrieved_chunks):
    """Generate an answer using retrieved context.

    Returns:
        tuple: (answer_text, cost)
    """
    # Build context from retrieved chunks
    context = ""
    for i, chunk in enumerate(retrieved_chunks, 1):
        context += f"\n[Source {i}: {chunk['source']}]\n{chunk['text']}\n"

    system_prompt = """You are a helpful AI assistant that answers questions based ONLY on the provided context.

Rules:
1. Use ONLY information from the provided sources
2. Cite sources explicitly (e.g., "According to Source 1...")
3. If the context doesn't contain enough information, say so clearly
4. Do not use your general knowledge - only use what's in the context
5. Keep answers concise and relevant"""

    user_prompt = f"""Context from news articles:
{context}

Question: {question}

Answer the question using only the information above. Cite your sources."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=1024,
    )

    cost = (
        response.usage.prompt_tokens * LLM_INPUT_COST_PER_TOKEN
        + response.usage.completion_tokens * LLM_OUTPUT_COST_PER_TOKEN
    )

    return response.choices[0].message.content, cost
