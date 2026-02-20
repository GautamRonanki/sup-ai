from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

LLM_INPUT_COST_PER_TOKEN = 0.15 / 1_000_000
LLM_OUTPUT_COST_PER_TOKEN = 0.60 / 1_000_000


def rewrite_query(question):
    """Use LLM to clean and expand the user's question before retrieval."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a search query optimizer. Your job is to:
1. Fix any spelling mistakes - if a word doesn't make sense, assume it's a typo and correct it to the closest real word
2. For unclear proper nouns or brand names that look misspelled, make your best guess at the correct spelling
3. Rephrase the question to be clearer for semantic search

Return ONLY the rewritten query. Nothing else. No explanation.""",
            },
            {"role": "user", "content": f"Rewrite this search query: {question}"},
        ],
        temperature=0,
        max_tokens=100,
    )

    cost = (
        response.usage.prompt_tokens * LLM_INPUT_COST_PER_TOKEN
        + response.usage.completion_tokens * LLM_OUTPUT_COST_PER_TOKEN
    )

    rewritten = response.choices[0].message.content.strip()
    return rewritten, cost
