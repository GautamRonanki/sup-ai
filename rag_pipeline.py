from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# Pricing per token
LLM_INPUT_COST_PER_TOKEN = 0.15 / 1_000_000  # gpt-4o-mini input
LLM_OUTPUT_COST_PER_TOKEN = 0.60 / 1_000_000  # gpt-4o-mini output


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


def classify_generation(question, retrieved_chunks, answer):
    """Use LLM-as-a-judge to classify generation quality."""

    context_preview = "\n".join([c["text"][:200] for c in retrieved_chunks])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are an answer quality classifier. Given a question, the context provided to an AI, and the AI's answer, classify the answer as one of:

- "refused": the AI said it doesn't have information, cannot answer, or the context doesn't contain relevant info
- "hedged": the AI gave a partial answer but expressed uncertainty or said information was limited
- "confident": the AI gave a direct, complete answer

Reply with ONLY a JSON object like this:
{"status": "refused", "reason": "brief explanation"}""",
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nContext given to AI:\n{context_preview}\n\nAI Answer: {answer}",
            },
        ],
        temperature=0,
        max_tokens=100,
    )

    import json

    try:
        raw = response.choices[0].message.content.strip()
        if not raw:
            return {"status": "unknown", "reason": "empty response from classifier"}
        result = json.loads(raw)
        return result
    except Exception as e:
        return {"status": "unknown", "reason": f"classifier failed: {str(e)}"}


def handle_refusal(question, retrieved_chunks):
    """Generate a helpful redirect when retrieval fails."""

    context_preview = "\n".join([c["text"][:300] for c in retrieved_chunks])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a helpful assistant. The user asked a question that cannot be answered from the available sources. 
Your job is to:
1. Politely tell them their question can't be answered from the loaded sources
2. In one sentence, describe what the sources actually seem to be about
3. Suggest a more relevant question they could ask

Keep it short, friendly, and helpful.""",
            },
            {
                "role": "user",
                "content": f"User question: {question}\n\nAvailable context:\n{context_preview}",
            },
        ],
        temperature=0.3,
        max_tokens=150,
    )

    return response.choices[0].message.content.strip()
