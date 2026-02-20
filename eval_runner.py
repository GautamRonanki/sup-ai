import json
import os
import time
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv

from retrieval import retrieve_relevant_chunks, classify_retrieval
from rag_pipeline import generate_answer, classify_generation, handle_refusal
from query_rewriter import rewrite_query

load_dotenv()
client = OpenAI()

# ==========================================
# CONFIG
# ==========================================
EVAL_SET_FILE = "eval_set.json"
RESULTS_FILE = "eval_results.json"
SOURCE_URL = "https://en.wikipedia.org/wiki/States_and_union_territories_of_India"

EMBED_COST_PER_TOKEN = 0.02 / 1_000_000  # text-embedding-3-small
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 50  # overlap between chunks


# ==========================================
# STEP 1: SCRAPE
# ==========================================
def scrape_url(url):
    """Scrape plain text from a URL."""
    print(f"Scraping: {url}")
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


# ==========================================
# STEP 2: CHUNK
# ==========================================
def chunk_text(text, source, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size
        chunk_body = text[start:end]

        if chunk_body.strip():
            chunks.append(
                {
                    "text": chunk_body,
                    "source": source,
                    "chunk_id": chunk_id,
                }
            )
            chunk_id += 1

        start += chunk_size - overlap

    print(f"Created {len(chunks)} chunks")
    return chunks


# ==========================================
# STEP 3: EMBED + BUILD INDEX
# ==========================================
def build_index(chunks):
    """Embed all chunks and build the index_data list."""
    print(f"Embedding {len(chunks)} chunks...")
    index_data = []
    total_cost = 0.0

    for i, chunk in enumerate(chunks):
        response = client.embeddings.create(
            input=chunk["text"], model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        cost = response.usage.total_tokens * EMBED_COST_PER_TOKEN
        total_cost += cost

        index_data.append(
            {
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "embedding": embedding,
            }
        )

        if (i + 1) % 20 == 0:
            print(f"  Embedded {i + 1}/{len(chunks)} chunks...")

    print(f"Index built. Embedding cost: ${total_cost:.4f}")
    return index_data, total_cost


# ==========================================
# STEP 4: RUN EVAL
# ==========================================
def run_evaluation(index_data, eval_set):
    """Run each test case through the pipeline and collect results."""
    results = []
    total_cost = 0.0

    for i, test_case in enumerate(eval_set):
        print(f"\n[{i + 1}/{len(eval_set)}] Q: {test_case['question']}")

        # Query rewriting
        start = time.time()
        rewritten_query, rewrite_cost = rewrite_query(test_case["question"])
        rewrite_time = time.time() - start
        total_cost += rewrite_cost

        # Retrieval
        start = time.time()
        chunks, retrieval_cost = retrieve_relevant_chunks(
            rewritten_query, index_data, top_k=3
        )
        retrieval_time = time.time() - start
        total_cost += retrieval_cost

        # Classification
        retrieval_class = classify_retrieval(chunks)

        # Generation — default to 0 in case retrieval failed
        generation_time = 0.0
        if retrieval_class["status"] == "failed":
            actual_answer = handle_refusal(test_case["question"], chunks)
        else:
            start = time.time()
            actual_answer, llm_cost = generate_answer(test_case["question"], chunks)
            generation_time = time.time() - start
            total_cost += llm_cost

        generation_class = classify_generation(
            test_case["question"], chunks, actual_answer
        )

        # Score using LLM-as-a-judge
        score, score_reason = score_answer(
            question=test_case["question"],
            expected_answer=test_case["expected_answer"],
            actual_answer=actual_answer,
            should_answer=test_case["should_answer"],
        )

        result = {
            "id": test_case["id"],
            "question": test_case["question"],
            "expected_answer": test_case["expected_answer"],
            "should_answer": test_case["should_answer"],
            "rewritten_query": rewritten_query,
            "actual_answer": actual_answer,
            "retrieval_status": retrieval_class["status"],
            "retrieval_score": retrieval_class["top_score"],
            "generation_status": generation_class["status"],
            "score": score,
            "score_reason": score_reason,
            "latency": {
                "rewrite_seconds": round(rewrite_time, 3),
                "retrieval_seconds": round(retrieval_time, 3),
                "generation_seconds": round(generation_time, 3),
                "total_seconds": round(
                    rewrite_time + retrieval_time + generation_time, 3
                ),
            },
        }

        results.append(result)
        print(f"  Score: {score}/5 — {score_reason}")
        print(
            f"  Latency: rewrite={rewrite_time:.2f}s | retrieval={retrieval_time:.2f}s | generation={generation_time:.2f}s"
        )

        time.sleep(0.5)

    print(f"\nTotal eval cost: ${total_cost:.4f}")
    return results, total_cost


# ==========================================
# STEP 5: SCORE WITH LLM-AS-A-JUDGE
# ==========================================
def score_answer(question, expected_answer, actual_answer, should_answer):
    """Use LLM-as-a-judge to score the actual answer against expected."""

    if should_answer:
        prompt = f"""You are an answer quality judge. Score the AI's answer from 1 to 5.

Question: {question}
Expected answer: {expected_answer}
AI's actual answer: {actual_answer}

Scoring guide:
5 - Correct and complete, matches expected answer
4 - Mostly correct, minor details missing
3 - Partially correct, key facts present but incomplete
2 - Mostly wrong or missing key facts
1 - Completely wrong or hallucinated

Reply ONLY with a JSON object like:
{{"score": 4, "reason": "brief explanation"}}"""
    else:
        prompt = f"""You are an answer quality judge. The AI should have REFUSED to answer this question because it's outside the knowledge base.

Question: {question}
AI's actual answer: {actual_answer}

Score the AI's response from 1 to 5:
5 - Correctly refused and explained it doesn't have this information
4 - Refused but explanation was vague
3 - Partially refused but gave some irrelevant answer
2 - Tried to answer despite not having the information
1 - Confidently gave a wrong answer (hallucinated)

Reply ONLY with a JSON object like:
{{"score": 4, "reason": "brief explanation"}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100,
    )

    try:
        raw = response.choices[0].message.content.strip()
        result = json.loads(raw)
        return result["score"], result["reason"]
    except Exception as e:
        return 0, f"scoring failed: {str(e)}"


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    # Load eval set
    with open(EVAL_SET_FILE, "r") as f:
        eval_set = json.load(f)
    print(f"Loaded {len(eval_set)} test cases")

    # Build index
    raw_text = scrape_url(SOURCE_URL)
    chunks = chunk_text(raw_text, source=SOURCE_URL)
    index_data, embed_cost = build_index(chunks)

    # Run evaluation
    results, eval_cost = run_evaluation(index_data, eval_set)

    # Summary stats
    scores = [r["score"] for r in results]
    avg_score = sum(scores) / len(scores)

    all_latencies = [r["latency"]["total_seconds"] for r in results]
    avg_latency = sum(all_latencies) / len(all_latencies)

    retrieval_latencies = [r["latency"]["retrieval_seconds"] for r in results]
    generation_latencies = [
        r["latency"]["generation_seconds"]
        for r in results
        if r["latency"]["generation_seconds"] > 0
    ]

    # Save results
    output = {
        "source": SOURCE_URL,
        "total_test_cases": len(results),
        "total_cost": round(embed_cost + eval_cost, 6),
        "summary": {
            "average_score": round(avg_score, 2),
            "scores": scores,
            "avg_total_latency_seconds": round(avg_latency, 3),
            "avg_retrieval_latency_seconds": round(
                sum(retrieval_latencies) / len(retrieval_latencies), 3
            ),
            "avg_generation_latency_seconds": round(
                sum(generation_latencies) / len(generation_latencies), 3
            ),
        },
        "results": results,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"Average score: {avg_score:.1f}/5")
    print(f"Average total latency: {avg_latency:.2f}s")
    print(
        f"Average retrieval latency: {sum(retrieval_latencies) / len(retrieval_latencies):.2f}s"
    )
    print(
        f"Average generation latency: {sum(generation_latencies) / len(generation_latencies):.2f}s"
    )
