import logging
import re
from pathlib import Path
from typing import List, Dict
from rag.logging_config import setup_logger
from rag.vector_store import dimension, COLLECTION_NAME, client
from sentence_transformers import CrossEncoder

logger = setup_logger(__name__)

def _resolve_ms_marco_model_path() -> str:
    return "cross-encoder/ms-marco-MiniLM-L-6-v2"

reranker = CrossEncoder(_resolve_ms_marco_model_path())

def search(query: str, query_embedding: List[float], k: int = 5, alpha: float = 0.7, initial_k: int = 20) -> List[Dict[str, str]]:
    if not query_embedding:
        logger.warning("Empty query embedding provided for hybrid search.")
        return []
    if len(query_embedding) != dimension:
        logger.error(f"Query embedding dimension mismatch. Expected {dimension}, got {len(query_embedding)}.")
        return []
    try:
        semantic_results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=initial_k,
            with_payload=True,
            with_vectors=False,
        )
        semantic_hits = [
            {
                "text": hit.payload.get("text", ""),
                "file": hit.payload.get("source_file", "unknown"),
                "semantic_score": hit.score
            }
            for hit in semantic_results if hit.payload and "text" in hit.payload
        ]
        query_terms = re.findall(r"\w+", query.lower())
        for hit in semantic_hits:
            text_lower = hit["text"].lower()
            keyword_score = sum(text_lower.count(term) for term in query_terms)
            hit["keyword_score"] = keyword_score
            hit["combined_score"] = alpha * hit["semantic_score"] + (1 - alpha) * keyword_score
        pre_ranked = sorted(semantic_hits, key=lambda x: x["combined_score"], reverse=True)
        pairs = [(query, doc["text"]) for doc in pre_ranked]
        rerank_scores = reranker.predict(pairs)
        for doc, score in zip(pre_ranked, rerank_scores):
            doc["rerank_score"] = float(score)
        final_ranked = sorted(pre_ranked, key=lambda x: x["rerank_score"], reverse=True)
        return [
            {
                "text": r["text"],
                "file": r["file"],
                "score": round(r["rerank_score"], 4)
            }
            for r in final_ranked[:k]
        ]
    except Exception as e:
        logger.exception(f"Error during hybrid search with rerank: {e}")
        return []
