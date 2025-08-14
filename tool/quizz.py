import json
import re
from typing import List, Dict, Any
import os
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate

# Initialize Groq LLM
def initialize_groq_llm(api_key: str = None, model: str = "deepseek-r1-distill-llama-70b"):
    """Initialize Groq LLM with API key."""
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set or api_key parameter not provided")
    
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model,
        temperature=0.3,  # Lower temperature for more consistent quiz generation
        max_tokens=2048
    )

def create_quizz(conversation: str, api_key: str = None) -> str:
    """
    Takes a conversation/chat history and returns 5 quiz questions with multiple options,
    correct answers, and explanations in JSON format using LangChain Groq.
    
    Args:
        conversation (str): The conversation or chat history to analyze
        api_key (str, optional): Groq API key. If not provided, uses GROQ_API_KEY env var
        
    Returns:
        str: JSON string containing quiz questions
    """
    
    try:
        # Initialize Groq LLM
        llm = initialize_groq_llm(api_key)
        
        # Clean and preprocess the conversation
        cleaned_text = _preprocess_conversation(conversation)
        
        # Generate quiz using LLM
        quiz_data = _generate_quiz_with_llm(llm, cleaned_text)
        
        return json.dumps(quiz_data, indent=2, ensure_ascii=False)
        
    except Exception as e:
        # Fallback to rule-based approach if LLM fails
        print(f"LLM generation failed: {e}. Falling back to rule-based approach.")
        return _fallback_quiz_generation(conversation)

def _preprocess_conversation(conversation: str) -> str:
    """Clean and preprocess the conversation text."""
    # Remove extra whitespace and normalize
    cleaned = re.sub(r'\s+', ' ', conversation.strip())
    
    # Remove common chat artifacts if present
    cleaned = re.sub(r'^\w+:\s*', '', cleaned, flags=re.MULTILINE)
    
    # Remove URLs and special characters that might confuse the LLM
    cleaned = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', cleaned)
    
    return cleaned

def _generate_quiz_with_llm(llm: ChatGroq, text: str) -> Dict[str, Any]:
    """Generate quiz using LangChain Groq LLM."""
    
    # Create a comprehensive prompt for quiz generation
    quiz_prompt = PromptTemplate(
        input_variables=["conversation_text"],
        template="""
You are an expert quiz generator. Based on the following conversation/text, create exactly 5 high-quality multiple-choice questions that test comprehension and understanding of the key concepts discussed.

CONVERSATION TEXT:
{conversation_text}

REQUIREMENTS:
1. Generate exactly 5 questions
2. Each question should have 4 multiple choice options (A, B, C, D)
3. Include the correct answer (A, B, C, or D)
4. Provide a clear explanation for the correct answer
5. Questions should cover different aspects: facts, concepts, relationships, implications, and analysis
6. Avoid trivial questions - focus on meaningful content
7. Make incorrect options plausible but clearly wrong
8. Ensure questions are directly answerable from the conversation content

OUTPUT FORMAT (JSON):
{{
  "quiz_title": "Conversation Comprehension Quiz",
  "total_questions": 5,
  "questions": [
    {{
      "id": 1,
      "question": "Question text here?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer": "A",
      "explanation": "Detailed explanation of why this answer is correct"
    }}
  ]
}}

Generate the quiz now:
"""
    )
    
    # Create the prompt
    formatted_prompt = quiz_prompt.format(conversation_text=text[:4000])  # Limit text length
    
    # Generate quiz using LLM
    messages = [
        SystemMessage(content="You are an expert quiz generator."),
        HumanMessage(content=formatted_prompt)
    ]
    
    response = llm.invoke(messages)
    response_text = response.content
    
    try:
        # Find the JSON part of the response
        json_match = re.search(r'```json\n({.*?})\n```', response_text, re.DOTALL)
        if not json_match:
            json_match = re.search(r'({.*?})', response_text, re.DOTALL)
        
        if json_match:
            quiz_json_str = json_match.group(1)
            quiz_data = json.loads(quiz_json_str)
            return _validate_and_clean_quiz(quiz_data)
        else:
            # If no JSON is found, try to create a fallback from the raw text
            return _create_fallback_from_llm_response(response_text, text)
            
    except (json.JSONDecodeError, KeyError):
        # If JSON parsing fails, create a fallback from the raw text
        return _create_fallback_from_llm_response(response_text, text)

def _validate_and_clean_quiz(quiz_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and clean the quiz data structure."""
    
    # Ensure required fields exist
    if "questions" not in quiz_data:
        quiz_data["questions"] = []
    
    if "quiz_title" not in quiz_data:
        quiz_data["quiz_title"] = "Conversation Comprehension Quiz"
    
    # Clean and validate questions
    valid_questions = []
    for i, question in enumerate(quiz_data["questions"][:5]):  # Limit to 5 questions
        if isinstance(question, dict):
            cleaned_question = {
                "id": question.get("id", i + 1),
                "question": str(question.get("question", f"Question {i + 1}")).strip(),
                "options": question.get("options", ["Option A", "Option B", "Option C", "Option D"])[:4],
                "correct_answer": str(question.get("correct_answer", "A")).upper(),
                "explanation": str(question.get("explanation", "No explanation provided")).strip()
            }
            
            # Ensure we have 4 options
            while len(cleaned_question["options"]) < 4:
                cleaned_question["options"].append(f"Option {chr(65 + len(cleaned_question['options']))}")
            
            # Validate correct answer
            if cleaned_question["correct_answer"] not in ["A", "B", "C", "D"]:
                cleaned_question["correct_answer"] = "A"
            
            valid_questions.append(cleaned_question)
    
    quiz_data["questions"] = valid_questions
    quiz_data["total_questions"] = len(valid_questions)
    
    return quiz_data

def _create_fallback_from_llm_response(response_text: str, original_text: str) -> Dict[str, Any]:
    """Create a structured quiz when JSON parsing fails but LLM provided content."""
    
    # Try to extract questions from the response text
    questions = []
    question_pattern = r'(\d+\.?\s*)(.*?\?)'
    matches = re.findall(question_pattern, response_text, re.DOTALL)
    
    for i, (num, question_text) in enumerate(matches[:5]):
        question = {
            "id": i + 1,
            "question": question_text.strip(),
            "options": [
                f"Based on the conversation content",
                f"Not mentioned in the conversation", 
                f"Opposite of what was discussed",
                f"Partially correct but incomplete"
            ],
            "correct_answer": "A",
            "explanation": "This answer is derived from the conversation content."
        }
        questions.append(question)
    
    # If no questions found, create generic ones
    if not questions:
        questions = _create_generic_questions(original_text)
    
    return {
        "quiz_title": "Conversation Comprehension Quiz",
        "total_questions": len(questions),
        "questions": questions
    }

def _create_generic_questions(text: str) -> List[Dict[str, Any]]:
    """Create generic questions when all else fails."""
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 30][:5]
    questions = []
    
    for i, sentence in enumerate(sentences):
        words = sentence.split()
        key_concept = ' '.join(words[:3]) if len(words) >= 3 else "the topic"
        
        question = {
            "id": i + 1,
            "question": f"What was mentioned about {key_concept}?",
            "options": [
                sentence[:70] + "..." if len(sentence) > 70 else sentence,
                "This was not discussed",
                "The opposite was stated",
                "It was mentioned but unclear"
            ],
            "correct_answer": "A",
            "explanation": f"The conversation included: '{sentence[:100]}...'" if len(sentence) > 100 else f"The conversation included: '{sentence}'"
        }
        questions.append(question)
    
    return questions

def _fallback_quiz_generation(conversation: str) -> str:
    """Fallback to rule-based quiz generation if LLM fails."""
    # Use the original rule-based approach as fallback
    cleaned_text = _preprocess_conversation(conversation)
    key_topics = _extract_key_topics_fallback(cleaned_text)
    questions = _generate_questions_fallback(cleaned_text, key_topics)
    
    quiz_data = {
        "quiz_title": "Conversation Quiz (Fallback Mode)",
        "total_questions": len(questions),
        "questions": questions
    }
    
    return json.dumps(quiz_data, indent=2, ensure_ascii=False)

def _extract_key_topics_fallback(text: str) -> List[str]:
    """Fallback topic extraction."""
    sentences = text.split('.')
    key_topics = []
    
    important_keywords = [
        'because', 'therefore', 'however', 'important', 'key', 'main', 
        'significant', 'result', 'conclusion', 'definition', 'means'
    ]
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 20:
            for keyword in important_keywords:
                if keyword.lower() in sentence.lower():
                    key_topics.append(sentence)
                    break
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 50 and len(sentence) < 200:
            key_topics.append(sentence)
    
    return key_topics[:10]

def _generate_questions_fallback(text: str, key_topics: List[str]) -> List[Dict[str, Any]]:
    """Fallback question generation."""
    questions = []
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
    
    for i, topic in enumerate(key_topics[:5]):
        question = {
            "id": i + 1,
            "question": f"What was discussed in the conversation regarding this topic?",
            "options": [
                topic[:80] + "..." if len(topic) > 80 else topic,
                "This was not mentioned",
                "The opposite was discussed", 
                "It was unclear from the conversation"
            ],
            "correct_answer": "A",
            "explanation": f"The conversation mentioned: '{topic}'"
        }
        questions.append(question)
    
    return questions

# Example usage
if __name__ == "__main__":
    # Set your Groq API key as environment variable or pass it directly
    # os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"
    
    sample_conversation = """
    Climate change is a significant global issue that affects weather patterns worldwide. 
    Rising temperatures cause ice caps to melt, which leads to sea level rise. 
    Renewable energy sources like solar and wind power can help reduce carbon emissions. 
    However, the transition to clean energy requires substantial investment and political will. 
    Scientists agree that immediate action is necessary to prevent catastrophic changes.
    The Paris Agreement aims to limit global warming to well below 2 degrees Celsius.
    Carbon pricing mechanisms can incentivize businesses to reduce their emissions.
    """
    
    try:
        quiz_json = create_quizz(sample_conversation)
        print(quiz_json)
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set your GROQ_API_KEY environment variable")