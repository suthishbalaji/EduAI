from rag.logging_config import setup_logger
import atexit
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from typing import List, Dict
import uuid

logger = setup_logger(__name__)
client = QdrantClient(path="qdrant_data")  

# Ensure the embedded Qdrant instance closes cleanly before Python shutdown
def _close_qdrant_client() -> None:
    try:
        client.close()
    except Exception:
        # Best-effort close; ignore errors during interpreter teardown
        pass

atexit.register(_close_qdrant_client)

dimension = 384
COLLECTION_NAME = "documents"


def init_qdrant() -> None:
    logger.info(f"QdrantClient initialized (persistent). Collection: {COLLECTION_NAME}")
    try:
        existing_collections = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME not in existing_collections:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
            logger.info(f"Qdrant collection '{COLLECTION_NAME}' created.")
        else:
            logger.info(f"Qdrant collection '{COLLECTION_NAME}' already exists. Skipping creation.")
    except Exception as e:
        logger.exception(f"Error initializing Qdrant collection: {e}")
        raise

def store_vectors(chunks: List[str], embeddings: List[List[float]], metadata: Dict) -> None:
    if not embeddings or not chunks:
        logger.warning("No chunks or embeddings provided for storage.")
        return

    if not isinstance(metadata, dict):
        logger.error(f"Metadata must be a dictionary, got {type(metadata).__name__}")
        return

    valid_points: List[PointStruct] = []
    for text, vec in zip(chunks, embeddings):
        if len(vec) != dimension:
            logger.warning(
                f"Skipping chunk due to embedding size mismatch: expected {dimension}, got {len(vec)}"
            )
            continue
        valid_points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={**metadata, "text": text}  
            )
        )

    if not valid_points:
        logger.warning("No valid vectors to upsert into Qdrant.")
        return

    try:
        client.upsert(collection_name=COLLECTION_NAME, points=valid_points, wait=True)
        logger.info(
            f"Successfully upserted {len(valid_points)} vectors with metadata={metadata} into Qdrant collection"
        )
    except Exception as e:
        logger.exception(f"Failed to upsert vectors into Qdrant: {e}")
