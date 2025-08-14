import os
from rag.logging_config import setup_logger
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

load_dotenv()

logger = setup_logger(__name__)

GROQ_API_KEY_RAG = os.getenv("GROQ_API_KEY_RAG")
if not GROQ_API_KEY_RAG:
    logger.error("GROQ_API_KEY not set in environment.")
    raise RuntimeError("GROQ_API_KEY not found.")


try:
    llm = ChatGroq(
        api_key=GROQ_API_KEY_RAG,
        model="llama-3.3-70b-versatile", 
        temperature=0.7,
        max_tokens=500
    )
    logger.info("Using LangChain ChatGroq model.")
except Exception as e:
    logger.exception("Failed to initialize ChatGroq.")
    raise e

def generate_answer_hyde(question: str) -> str:
    if not question.strip():
        logger.warning("Empty question provided.")
        return "Error: Question is missing."

    messages = [
        SystemMessage(content=(
            "You are an expert researcher. Given a question, write a detailed, "
            "well-structured, and informative answer that could plausibly appear "
            "in a real document. Do not mention that this is hypothetical."
        )),
        HumanMessage(content=f"Question:\n{question}\n\nWrite a detailed answer.")
    ]

    try:
        logger.debug(f"Calling Groq with question: {question}")
        response = llm.invoke(messages)  
        answer = response.content.strip()

        logger.info("Hypothetical document generated successfully with Groq.")
        return answer

    except Exception as e:
        logger.exception("Groq API call for HYDE failed.")
        return f"Error: Failed to generate hypothetical document due to API error. {str(e)}"
