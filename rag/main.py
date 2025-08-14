import os
from dotenv import load_dotenv
from rag.parser import _extract_pdf_text
from rag.chunker import chunk_text
from rag.embedder import embed_chunks,embed_query_hyde
from rag.vector_store import init_qdrant, store_vectors
from rag.logging_config import setup_logger
from rag.retriever import search
load_dotenv()
logger = setup_logger(__name__)
import uuid
try:
    init_qdrant()
    logger.info("Qdrant collection initialized successfully.")
except Exception:
    logger.exception("Failed to initialize Qdrant.")
    raise


def process_document(pdf_file) -> str:
    try:
        doc_id = uuid.uuid4()
        logger.info(f"Starting PDF processing for document ID: {doc_id}")

        if isinstance(pdf_file, str) and os.path.exists(pdf_file):
            with open(pdf_file, "rb") as f:
                content = f.read()
        else:
            content = pdf_file.read()

        if not content:
            logger.warning("No content found in the provided PDF file.")
            return ""

        text = _extract_pdf_text(content)
        if not text:
            logger.warning("No text extracted from PDF.")
            return ""

        logger.info(f"Extracted {len(text)} characters from PDF.")

        chunks = chunk_text(text)
        logger.info(f"Generated {len(chunks)} text chunks.")

        embeddings = embed_chunks(chunks)
        logger.info(f"Generated {len(embeddings)} embeddings.")

        store_vectors(chunks, embeddings, metadata={"doc_id": str(doc_id)})
        logger.info(f"Document chunks stored in vector store with doc_id={doc_id}.")

        return f"Document '{doc_id}' processed and stored successfully."

    except Exception as e:
        logger.exception(f"[ERROR] Failed in process_document: {e}")
        return ""


def process_query(query: str):
    try:
        logger.info("Processing query")

        query_vec = embed_query_hyde(query)
        if not query_vec:
            logger.warning("No embedding generated for query.")
            return {"context": ""}

        top_chunks = search(query, query_vec, k=5, alpha=0.7)
        if not top_chunks:
            logger.warning("No results retrieved for query.")
            return {"context": ""}

        context = "\n".join(chunk["text"] for chunk in top_chunks)
        return {"context": context}

    except Exception as e:
        logger.exception(f"[ERROR] Failed in process_query: {e}")
        return {"context": ""}
