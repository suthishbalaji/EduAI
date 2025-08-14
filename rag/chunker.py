from langchain.text_splitter import RecursiveCharacterTextSplitter
from rag.logging_config import setup_logger

logger = setup_logger(__name__)


def chunk_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 200) -> list:
    try:
        logger.info("Started chunking the text")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_text(text)
        logger.info("Successfully chunked the text")
        return chunks

    except Exception as e:
        logger.exception(f"[ERROR] Failed to split text: {e}")
        return []
