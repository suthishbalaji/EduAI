from rag.logging_config import setup_logger
from sentence_transformers import SentenceTransformer
from typing import List
from rag.llm import generate_answer_hyde

logger = setup_logger(__name__)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_chunks(texts: List[str]) -> List[List[float]]:
    if not texts:
        logger.warning("No texts provided for embedding.")
        return []
    try:
        embeddings = model.encode(texts).tolist()
        logger.info(f"Generated {len(embeddings)} embeddings.")
        return embeddings
    except Exception as e:
        logger.exception(f"Error generating embeddings: {e}")
        return []
    


def embed_query_hyde(query: str) -> List[float]:
    try:
        logger.info("Generating hypothetical document for HYDE embedding.")

       
        hypothetical_doc = generate_answer_hyde(question=query)

        if not hypothetical_doc.strip():
            logger.warning("Generated hypothetical document is empty.")
            return []

        
        embedding = model.encode([hypothetical_doc])[0].tolist()

        logger.info("HYDE embedding successfully created.")
        return embedding

    except Exception as e:
        logger.exception(f"Error creating HYDE embedding: {e}")
        return []