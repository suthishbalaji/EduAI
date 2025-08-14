import os
from rag.logging_config import setup_logger
from io import BytesIO

logger = setup_logger(__name__)

def _extract_pdf_text(content: bytes) -> str:
    try:
        import PyPDF2
    except ImportError:
        logger.error("PyPDF2 not installed. Please add it to requirements to parse PDFs.")
        return ""

   
    text = ""
    try:
        reader = PyPDF2.PdfReader(BytesIO(content))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        text = "\n".join(pages).strip()
    except Exception as e:
        logger.exception(f"[ERROR] Failed to read PDF: {e}")
        text = ""

    
    if not text.strip():
        logger.warning("No text found using PyPDF2, falling back to OCR...")
        try:
            from pdf2image import convert_from_bytes
            import pytesseract

            images = convert_from_bytes(content)
            ocr_text = []
            for img in images:
                ocr_text.append(pytesseract.image_to_string(img))
            text = "\n".join(ocr_text).strip()
        except ImportError:
            logger.error("OCR fallback failed. Please install: pdf2image, pytesseract, and Tesseract OCR.")
        except Exception as e:
            logger.exception(f"[ERROR] OCR extraction failed: {e}")

    return text
