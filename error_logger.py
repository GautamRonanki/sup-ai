import json
import os
from datetime import datetime

LOG_FILE = "error_log.json"


def log_query(
    question,
    rewritten_query,
    retrieval_classification,
    generation_classification,
    top_chunks,
    answer,
):
    """Log every query with its error classification for diagnosis."""

    entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "rewritten_query": rewritten_query,
        "retrieval": {
            "status": retrieval_classification["status"],
            "reason": retrieval_classification["reason"],
            "top_score": retrieval_classification["top_score"],
            "sources_retrieved": [c["source"] for c in top_chunks],
            "scores_per_source": [
                {"source": c["source"], "score": round(c["similarity"], 4)}
                for c in top_chunks
            ],
        },
        "generation": {
            "status": generation_classification["status"],
            "reason": generation_classification["reason"],
        },
        "overall_failure_type": _classify_overall(
            retrieval_classification, generation_classification
        ),
        "answer_preview": answer[:200],
    }

    # Load existing log or start fresh
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            log = json.load(f)
    else:
        log = []

    log.append(entry)

    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)

    return entry


def _classify_overall(retrieval_result, generation_result):
    """Determine the primary failure type."""
    if retrieval_result["status"] == "failed":
        return "retrieval_failure"
    elif (
        retrieval_result["status"] == "uncertain"
        and generation_result["status"] == "refused"
    ):
        return "retrieval_failure"
    elif (
        retrieval_result["status"] == "confident"
        and generation_result["status"] == "refused"
    ):
        return "generation_failure"
    elif generation_result["status"] == "hedged":
        return "generation_uncertain"
    else:
        return "none"
