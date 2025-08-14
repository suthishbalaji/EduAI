from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0.5,
    max_tokens=2048,
    groq_api_key=os.getenv("GROQ_API_KEY") 
)

def format_response(response_text):

    if not response_text:
        return "I apologize, but I couldn't generate a response. Please try again."
    
   
    import re
    
   
    response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL | re.IGNORECASE)
    
    
    unwanted_tags = ['<thinking>', '</thinking>', '<thought>', '</thought>']
    for tag in unwanted_tags:
        response_text = response_text.replace(tag, '')
    

    response_text = re.sub(r'\n\s*\n\s*\n', '\n\n', response_text) 
    response_text = response_text.strip()
    

    if not response_text.strip():
        return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
    
    return response_text

def get_language_instruction(language):
    """Get language-specific instruction for the AI"""
    if language == "Tamil":
        return "IMPORTANT: You MUST respond in Tamil (தமிழ்). Use Tamil script only."
    elif language == "Hindi":
        return "IMPORTANT: You MUST respond in Hindi (हिंदी). Use Devanagari script only."
    else:  # Default to English
        return "You should respond in English."

def chat_llm(user_message, context="", chat_history=None, language="English"):

    try:
        # Get language-specific instruction
        language_instruction = get_language_instruction(language)
        
        if context:
            system_prompt = f"""You are StudyMate Pro, an AI-powered study assistant. You have access to uploaded document content to help answer questions.

Context from uploaded documents:
{context}

Instructions:
- Use the provided context to answer questions accurately
- If the question can be answered using the context, prioritize that information
- If the context doesn't contain relevant information, use your general knowledge
- Be helpful, clear, and educational in your responses
- Format your responses nicely with proper structure when needed
- Do not include any thinking process or internal reasoning in your response
- Provide only the final, clean answer

{language_instruction}
"""
        else:
            system_prompt = f"""You are StudyMate Pro, an AI-powered study assistant. Help users with their studies by:
- Answering questions clearly and educationally
- Providing explanations and examples when helpful
- Being encouraging and supportive
- Structuring responses in a readable format
- Do not include any thinking process or internal reasoning in your response
- Provide only the final, clean answer

{language_instruction}
"""

        
        messages = [SystemMessage(content=system_prompt)]
        
        
        if chat_history:
            recent_history = chat_history[-10:] if len(chat_history) > 10 else chat_history
            for msg in recent_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg['content']))
        

        messages.append(HumanMessage(content=user_message))

        response = llm.invoke(messages)

        formatted_response = format_response(response.content)
        return formatted_response
        
    except Exception as e:
        print(f"Error in chat_llm: {e}")
        # Return error message in the requested language
        if language == "Tamil":
            return f"மன்னிக்கவும், உங்கள் கோரிக்கையை செயலாக்கும் போது ஒரு பிழை ஏற்பட்டது. தயவுசெய்து மீண்டும் முயற்சிக்கவும். பிழை: {str(e)}"
        elif language == "Hindi":
            return f"मुझे खेद है, लेकिन आपके अनुरोध को प्रोसेस करते समय एक त्रुटि हुई। कृपया फिर से कोशिश करें। त्रुटि: {str(e)}"
        else:
            return f"I apologize, but I encountered an error while processing your request. Please try again. Error: {str(e)}"

def simple_chat_response(user_message, language="English"):
    """
    Simple chat function without context - for fallback
    """
    try:
        # Get language-specific instruction
        language_instruction = get_language_instruction(language)
        
        messages = [
            SystemMessage(content=f"""You are StudyMate Pro, a helpful AI study assistant. Provide clear, educational responses.
            
Instructions:
- Be helpful, clear, and educational
- Do not include any thinking process or internal reasoning in your response
- Provide only the final, clean answer

{language_instruction}"""),
            HumanMessage(content=user_message)
        ]
        
        response = llm.invoke(messages)
        
        formatted_response = format_response(response.content)
        return formatted_response
        
    except Exception as e:
        print(f"Error in simple_chat_response: {e}")
        # Return error message in the requested language
        if language == "Tamil":
            return "இப்போது இணைக்க சிரமம் ஏற்படுகிறது. தயவுசெய்து பின்னர் முயற்சிக்கவும்."
        elif language == "Hindi":
            return "अभी कनेक्ट करने में परेशानी हो रही है। कृपया बाद में कोशिश करें।"
        else:
            return "I'm having trouble connecting right now. Please try again later."