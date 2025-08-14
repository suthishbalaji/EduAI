import streamlit as st
import os
import json
from datetime import datetime
import random
import sys
from io import BytesIO
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import speech recognition for audio processing
try:
    import speech_recognition as sr
    import tempfile
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False

# Import text-to-speech for audio output
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# Import RAG and chat modules
try:
    from rag.main import process_document, process_query
    from tool.chat import chat_llm, simple_chat_response
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# Import summarizer
try:
    from tool.summarizer import summarize_chat_with_granite
    SUMMARIZER_AVAILABLE = True
except ImportError:
    SUMMARIZER_AVAILABLE = False

# PDF generation (reportlab)
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="StudyMate Pro",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- FILE PATHS ---
UPLOAD_FOLDER = "uploads"
HISTORY_FILE = "chat_history.json"

# --- SETUP ---
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- HELPER FUNCTIONS ---
def save_uploaded_file(uploaded_file):
    """Saves an uploaded file to the local UPLOAD_FOLDER."""
    file_path = os.path.join(
        UPLOAD_FOLDER,
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_file.name}"
    )
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def load_chat_history():
    """Loads chat history from a local JSON file."""
    if os.path.exists(HISTORY_FILE) and os.path.getsize(HISTORY_FILE) > 0:
        with open(HISTORY_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {"Chat 1": []}
    return {"Chat 1": []}

def save_chat_history(history):
    """Saves chat history to a local JSON file."""
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def format_markdown_response(response_text):
    """Additional formatting for better display in Streamlit"""
    if not response_text:
        return "No response generated."
    
    import re
    # Ensure proper spacing around headers
    response_text = re.sub(r'\n(#+\s)', r'\n\n\1', response_text)
    # Ensure proper spacing around lists
    response_text = re.sub(r'\n(\*\s|\d+\.\s)', r'\n\n\1', response_text)
    # Clean up excessive spacing
    response_text = re.sub(r'\n\s*\n\s*\n', '\n\n', response_text)
    
    return response_text.strip()

def clean_text_for_speech(text):
    """Clean text for better text-to-speech pronunciation"""
    import re
    
    # Remove markdown formatting
    text = re.sub(r'\\(.?)\\*', r'\1', text)  # Bold
    text = re.sub(r'\(.?)\*', r'\1', text)      # Italic
    text = re.sub(r'(.*?)', r'\1', text)        # Code
    text = re.sub(r'#{1,6}\s', '', text)          # Headers
    text = re.sub(r'^\-\s', '', text, flags=re.MULTILINE)  # List items
    text = re.sub(r'^\d+\.\s', '', text, flags=re.MULTILINE)  # Numbered lists
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'link', text)
    
    # Replace emojis and symbols with readable text
    emoji_replacements = {
        '📚': 'books',
        '👋': 'hello',
        '😊': 'smile',
        '✅': 'check',
        '❌': 'error',
        '💡': 'idea',
        '🤖': 'AI',
        '🧑‍💻': 'user',
        '📄': 'document',
        '🎤': 'microphone',
        '✨': 'sparkles',
        '🚀': 'rocket'
    }
    
    for emoji, replacement in emoji_replacements.items():
        text = text.replace(emoji, replacement)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def speak_text(text, message_id):
    """Convert text to speech using pyttsx3"""
    if not TTS_AVAILABLE:
        st.error("Text-to-speech not available. Install pyttsx3: pip install pyttsx3")
        return
    
    try:
        # Clean text for better speech
        clean_text = clean_text_for_speech(text)
        
        # Limit text length to prevent very long speech
        if len(clean_text) > 1000:
            clean_text = clean_text[:1000] + "... text truncated for speech"
        
        # Initialize TTS engine
        engine = pyttsx3.init()
        
        # Set properties (optional - adjust as needed)
        engine.setProperty('rate', 180)    # Speed of speech
        engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
        
        # Get available voices and set a more natural one if available
        voices = engine.getProperty('voices')
        if voices:
            # Try to find a female voice or use the second available voice
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
            else:
                # If no female voice, use the first available voice
                engine.setProperty('voice', voices[0].id)
        
        # Speak the text
        engine.say(clean_text)
        engine.runAndWait()
        engine.stop()
        
    except Exception as e:
        st.error(f"Error with text-to-speech: {str(e)}")

def transcribe_audio(audio_file):
    """Transcribe audio file to text using speech_recognition library"""
    if not SPEECH_RECOGNITION_AVAILABLE:
        return "❌ Speech recognition not available. Please install: pip install SpeechRecognition"
    
    try:
        # Initialize recognizer with adjusted settings
        r = sr.Recognizer()
        r.energy_threshold = 300
        r.pause_threshold = 0.8
        r.phrase_threshold = 0.3
        
        # Get audio bytes from the uploaded file
        if hasattr(audio_file, 'read'):
            # It's an UploadedFile object
            audio_bytes = audio_file.read()
            audio_file.seek(0)  # Reset file pointer
        else:
            # It's already bytes
            audio_bytes = audio_file
        
        # Create a temporary file to save the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        try:
            # Use speech recognition with the temporary file
            with sr.AudioFile(tmp_file_path) as source:
                # Adjust for ambient noise
                r.adjust_for_ambient_noise(source, duration=0.2)
                audio_data = r.record(source)
                
                # Try Google Speech Recognition first (most accurate)
                try:
                    text = r.recognize_google(audio_data, language='en-US')
                    return text.strip()
                except sr.UnknownValueError:
                    return "❌ Could not understand the audio. Please speak more clearly and try again."
                except sr.RequestError as e:
                    return f"❌ Speech recognition service error: {str(e)}. Check your internet connection."
                    
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
            
    except Exception as e:
        return f"❌ Error processing audio: {str(e)}"

def generate_intelligent_reply(user_message, has_documents=False, chat_history=None, language="English"):
    """Generate an intelligent reply using RAG if available, otherwise simple chat"""
    try:
        if RAG_AVAILABLE and has_documents:
            # Use RAG to get context from uploaded documents
            rag_result = process_query(user_message)
            context = rag_result.get("context", "")
            
            if context:
                response = chat_llm(user_message, context=context, chat_history=chat_history, language=language)
            else:
                response = chat_llm(user_message, chat_history=chat_history, language=language)
        elif RAG_AVAILABLE:
            # No documents uploaded, use simple chat
            response = simple_chat_response(user_message, language=language)
        else:
            # RAG not available, use fallback
            response = generate_auto_reply(user_message, language)
        
        return format_markdown_response(response)
            
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return generate_auto_reply(user_message, language)

def generate_auto_reply(user_message, language="English"):
    """Generate a simple, rule-based auto-reply as fallback"""
    msg_lower = user_message.lower()
    
    # Language-specific responses
    if language == "Tamil":
        if any(word in msg_lower for word in ["hello", "hi", "hey", "வணக்கம்"]):
            return random.choice([
                "வணக்கம்! இன்று உங்கள் படிப்பில் எப்படி உதவ முடியும்? 📚",
                "வணக்கம்! 👋 எந்த விषயத்தைப் பற்றி அறிய விரும்புகிறீர்கள்?",
                "வணக்கம்! சேர்ந்து கற்றுக்கொள்ள தயாரா?"
            ])
        elif any(word in msg_lower for word in ["pdf", "upload", "document", "ஆவணம்"]):
            return "நீங்கள் PDF ஆவணங்களை பக்கப்பட்டியைப் பயன்படுத்தி பதிவேற்றலாம். பதிவேற்றியவுடன், அவை பற்றிய கேள்விகளுக்கு நான் உதவ முடியும்! 📄"
        elif any(word in msg_lower for word in ["thank", "thanks", "நன்றி"]):
            return "மிக்க மகிழ்ச்சி! 😊 உங்கள் படிப்பில் உதவ நான் எப்போதும் இங்கே இருக்கிறேன்."
        elif "help" in msg_lower or "உதவி" in msg_lower:
            return "நான் இங்கே உதவ இருக்கிறேன்! நீங்கள் கேள்விகள் கேட்கலாம், ஆவணங்களைப் பதிவேற்றலாம் அல்லது படிப்பு தொடர்பான உரையாடல் நடத்தலாம். என்ன செய்ய விரும்புகிறீர்கள்?"
        elif any(word in msg_lower for word in ["bye", "goodbye", "பிரியாவிடை"]):
            return "பிரியாவிடை! மகிழ்ச்சியான படிப்பு மற்றும் அடுத்த முறை சந்திப்போம்! 👋✨"
        else:
            return random.choice([
                "அது சுவாரஸ்யமானது! அதைப் பற்றி மேலும் என்ன அறிய விரும்புகிறீர்கள்?",
                "அதில் உதவ விரும்புகிறேன். எந்த குறிப்பிட்ட அம்சத்தில் கவனம் செலுத்த விரும்புகிறீர்கள்?",
                "நல்ல கேள்வி! இந்த தலைப்பில் இதுவரை உங்கள் எண்ணங்கள் என்ன?",
                "அதைப் பற்றி ஆழமாகப் பார்ப்போம். என்ன மேலும் அறிய விரும்புகிறீர்கள்?"
            ])
    
    elif language == "Hindi":
        if any(word in msg_lower for word in ["hello", "hi", "hey", "नमस्ते"]):
            return random.choice([
                "नमस्ते! आज मैं आपकी पढ़ाई में कैसे मदद कर सकता हूँ? 📚",
                "नमस्ते! 👋 आप किस विषय के बारे में जानना चाहते हैं?",
                "नमस्ते! क्या आप साथ मिलकर कुछ सीखने के लिए तैयार हैं?"
            ])
        elif any(word in msg_lower for word in ["pdf", "upload", "document", "दस्तावेज़"]):
            return "आप साइडबार का उपयोग करके PDF दस्तावेज़ अपलोड कर सकते हैं। अपलोड करने के बाद, मैं उनके बारे में सवालों के जवाब देने में मदद कर सकता हूँ! 📄"
        elif any(word in msg_lower for word in ["thank", "thanks", "धन्यवाद"]):
            return "आपका स्वागत है! 😊 मैं आपकी पढ़ाई में मदद के लिए हमेशा यहाँ हूँ।"
        elif "help" in msg_lower or "मदद" in msg_lower:
            return "मैं यहाँ मदद करने के लिए हूँ! आप सवाल पूछ सकते हैं, दस्तावेज़ अपलोड कर सकते हैं, या पढ़ाई से संबंधित बातचीत कर सकते हैं। आप क्या करना चाहते हैं?"
        elif any(word in msg_lower for word in ["bye", "goodbye", "अलविदा"]):
            return "अलविदा! खुशी से पढ़ाई करें और अगली बार मिलते हैं! 👋✨"
        else:
            return random.choice([
                "यह दिलचस्प है! आप इसके बारे में और क्या जानना चाहते हैं?",
                "मैं इसमें आपकी मदद करना चाहता हूँ। आप किस विशेष पहलू पर ध्यान देना चाहते हैं?",
                "अच्छा सवाल! इस विषय पर अब तक आपके क्या विचार हैं?",
                "आइए इसमें और गहराई से जाते हैं। आप और क्या जानना चाहते हैं?"
            ])
    
    else:  # English (default)
        if any(word in msg_lower for word in ["hello", "hi", "hey"]):
            return random.choice([
                "Hello! How can I assist you with your studies today? 📚",
                "Hi there! 👋 What topic would you like to explore?",
                "Hey! Ready to dive into some learning together?"
            ])
        elif any(word in msg_lower for word in ["pdf", "upload", "document"]):
            return "You can upload PDF documents using the sidebar. Once uploaded, I can help answer questions about them! 📄"
        elif any(word in msg_lower for word in ["thank", "thanks"]):
            return "You're most welcome! 😊 I'm here whenever you need help with your studies."
        elif "help" in msg_lower:
            return "I'm here to help! You can ask me questions, upload documents for me to analyze, or just have a study-related conversation. What would you like to work on?"
        elif any(word in msg_lower for word in ["bye", "goodbye"]):
            return "Goodbye! Happy studying and see you next time! 👋✨"
        else:
            return random.choice([
                "That's interesting! Could you tell me more about what you'd like to explore?",
                "I'd love to help you with that. What specific aspect would you like to focus on?",
                "Great question! What are your thoughts on this topic so far?",
                "Let's dive deeper into that. What would you like to know more about?"
            ])

def chat_messages_to_text(messages):
    """Convert list of chat messages to plain text transcript."""
    lines = []
    for message in messages:
        role = "User" if message.get("role") == "user" else "Assistant"
        content = message.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)

def create_pdf_bytes(text, title=None):
    """Create a simple PDF bytes object from text using reportlab if available."""
    if not REPORTLAB_AVAILABLE:
        return b""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    textobject = c.beginText(50, height - 72)
    textobject.setFont("Times-Roman", 12)
    if title:
        textobject.textLine(title)
        textobject.textLine("")
    for line in text.split("\n"):
        textobject.textLine(line)
    c.drawText(textobject)
    c.showPage()
    c.save()
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

# --- SESSION STATE INITIALIZATION ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()
if "current_chat_key" not in st.session_state:
    all_keys = list(st.session_state.chat_history.keys())
    st.session_state.current_chat_key = all_keys[0] if all_keys else "Chat 1"
if "processed_documents" not in st.session_state:
    st.session_state.processed_documents = []
if "processing_status" not in st.session_state:
    st.session_state.processing_status = {}
if "voice_text" not in st.session_state:
    st.session_state.voice_text = ""
if "audio_processed" not in st.session_state:
    st.session_state.audio_processed = False
if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None
if "speaking_status" not in st.session_state:
    st.session_state.speaking_status = {}
if "last_summary" not in st.session_state:
    st.session_state.last_summary = ""
if "last_summary_chat_key" not in st.session_state:
    st.session_state.last_summary_chat_key = None

# --- SIDEBAR ---
with st.sidebar:
    st.title("🚀 StudyMate Pro")
    st.markdown("Your AI-powered study companion")
    
    # Show system status
    with st.expander("🔧 System Status", expanded=False):
        st.write("📊 *Components Status:*")
        st.write(f"🎤 Speech Recognition: {'✅ Available' if SPEECH_RECOGNITION_AVAILABLE else '❌ Not Available'}")
        st.write(f"🔊 Text-to-Speech: {'✅ Available' if TTS_AVAILABLE else '❌ Not Available'}")
        st.write(f"🧠 RAG System: {'✅ Available' if RAG_AVAILABLE else '❌ Not Available'}")
        st.write(f"🧾 Summarizer: {'✅ Available' if SUMMARIZER_AVAILABLE else '❌ Not Available'}")
        st.write(f"📄 PDF Export: {'✅ Available' if REPORTLAB_AVAILABLE else '❌ Not Available'}")
        
        if not SPEECH_RECOGNITION_AVAILABLE:
            st.code("pip install SpeechRecognition")
        if not TTS_AVAILABLE:
            st.code("pip install pyttsx3")
        if not SUMMARIZER_AVAILABLE:
            st.code("pip install transformers torch --upgrade")
        if not REPORTLAB_AVAILABLE:
            st.code("pip install reportlab")
    
    st.markdown("---")

    st.subheader("📂 Upload Materials")
    
    if not RAG_AVAILABLE:
        st.warning("⚠ Document processing not available. Some imports missing.")
    
    uploaded_files = st.file_uploader(
        "Upload your PDFs here", 
        type=["pdf"], 
        accept_multiple_files=True,
        disabled=not RAG_AVAILABLE
    )
    
    if uploaded_files and RAG_AVAILABLE:
        for uploaded_file in uploaded_files:
            file_key = f"{uploaded_file.name}_{uploaded_file.size}"
            
            if file_key not in st.session_state.processing_status:
                file_path = save_uploaded_file(uploaded_file)
                st.success(f"Uploaded: {os.path.basename(file_path)}", icon="✅")
                
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        uploaded_file.seek(0)
                        result = process_document(uploaded_file)
                        
                        if result:
                            st.session_state.processed_documents.append({
                                "name": uploaded_file.name,
                                "path": file_path,
                                "processed_at": datetime.now().isoformat()
                            })
                            st.session_state.processing_status[file_key] = "success"
                            st.success(f"✅ {uploaded_file.name} processed successfully!")
                        else:
                            st.session_state.processing_status[file_key] = "failed"
                            st.error(f"❌ Failed to process {uploaded_file.name}")
                    except Exception as e:
                        st.session_state.processing_status[file_key] = "failed"
                        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            else:
                status = st.session_state.processing_status[file_key]
                if status == "success":
                    st.info(f"📄 {uploaded_file.name} (processed)")
                else:
                    st.warning(f"⚠ {uploaded_file.name} (failed)")

    # Show processed documents
    if st.session_state.processed_documents:
        st.markdown("### 📚 Loaded Documents")
        for i, doc in enumerate(st.session_state.processed_documents, 1):
            st.text(f"{i}. {doc['name']}")
    
    st.markdown("---")

    st.subheader("💬 Chat Management")
    if st.button("➕ New Chat", use_container_width=True):
        new_chat_num = 1
        while f"Chat {new_chat_num}" in st.session_state.chat_history:
            new_chat_num += 1
        new_chat_key = f"Chat {new_chat_num}"
        st.session_state.chat_history[new_chat_key] = []
        st.session_state.current_chat_key = new_chat_key
        save_chat_history(st.session_state.chat_history)
        st.rerun()

    for chat_key in list(st.session_state.chat_history.keys()):
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.session_state.current_chat_key == chat_key:
                st.button(chat_key, key=f"select_{chat_key}", use_container_width=True, type="primary")
            else:
                if st.button(chat_key, key=f"select_{chat_key}", use_container_width=True):
                    st.session_state.current_chat_key = chat_key
                    st.rerun()
        with col2:
            if len(st.session_state.chat_history) > 1:
                if st.button("🗑", key=f"delete_{chat_key}", use_container_width=True, help=f"Delete {chat_key}"):
                    del st.session_state.chat_history[chat_key]
                    if st.session_state.current_chat_key == chat_key:
                        st.session_state.current_chat_key = list(st.session_state.chat_history.keys())[0]
                    save_chat_history(st.session_state.chat_history)
                    st.rerun()

# --- MAIN CHAT INTERFACE ---
header_col1, header_col2 = st.columns([8, 2])
with header_col1:
    st.header(f"💬 {st.session_state.current_chat_key}")
with header_col2:
    if st.button("🧾 Summarize", key="summarize_button", use_container_width=True, help="Summarize this chat and download"):
        messages = st.session_state.chat_history.get(st.session_state.current_chat_key, [])
        if not messages:
            st.warning("No messages to summarize in this chat.")
        else:
            transcript = chat_messages_to_text(messages)
            if SUMMARIZER_AVAILABLE:
                with st.spinner("Summarizing chat..."):
                    try:
                        summary_text = summarize_chat_with_granite(transcript)
                        st.session_state.last_summary = summary_text or ""
                        st.session_state.last_summary_chat_key = st.session_state.current_chat_key
                    except Exception as e:
                        st.error(f"Error during summarization: {e}")
            else:
                st.error("Summarizer not available. Please install required dependencies.")

# Show document status
if st.session_state.processed_documents:
    st.info(f"📚 {len(st.session_state.processed_documents)} document(s) loaded and ready for questions!", icon="ℹ")
elif RAG_AVAILABLE:
    st.info("💡 Upload PDF documents in the sidebar to unlock document-based Q&A!", icon="💡")

# Show summary and download options if available for this chat
if st.session_state.last_summary and st.session_state.last_summary_chat_key == st.session_state.current_chat_key:
    st.subheader("🧾 Chat Summary")
    st.markdown(st.session_state.last_summary)
    dl_col1, dl_col2 = st.columns([1, 1])
    with dl_col1:
        st.download_button(
            "Download .txt",
            data=st.session_state.last_summary.encode("utf-8"),
            file_name=f"{st.session_state.current_chat_key.replace(' ', '_').lower()}_summary.txt",
            mime="text/plain",
            use_container_width=True,
            key="download_txt_summary"
        )
    with dl_col2:
        if REPORTLAB_AVAILABLE:
            pdf_bytes = create_pdf_bytes(
                st.session_state.last_summary,
                title=f"{st.session_state.current_chat_key} - Summary"
            )
            st.download_button(
                "Download .pdf",
                data=pdf_bytes,
                file_name=f"{st.session_state.current_chat_key.replace(' ', '_').lower()}_summary.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="download_pdf_summary"
            )
        else:
            st.button("Download .pdf (unavailable)", disabled=True, use_container_width=True)
            st.caption("Install reportlab to enable PDF export.")

current_chat = st.session_state.chat_history.get(st.session_state.current_chat_key, [])

# Welcome message for new chats
if not current_chat:
    welcome_msg = """
    👋 *Welcome to StudyMate Pro!* 
    
    I'm your AI study companion. Here's what I can do:
    
    - 💬 Answer questions and help with learning
    - 📄 Analyze uploaded PDF documents  
    - 🎤 Listen to your voice messages
    - 🔊 Read responses aloud with text-to-speech
    - 🧠 Remember our conversation context
    - 🌍 Respond in multiple languages (English, Tamil, Hindi)
    
    *Get started:* Type a message, upload a document, or try voice input below!
    """
    st.markdown(welcome_msg)

# Display chat history
for i, message in enumerate(current_chat):
    avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
    
    if message["role"] == "assistant":
        # Create columns for assistant messages with speaker button
        col1, col2 = st.columns([10, 1])
        
        with col1:
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
        
        with col2:
            if TTS_AVAILABLE:
                # Create unique key for each message
                message_key = f"speak_{st.session_state.current_chat_key}_{i}"
                
                # Speaker button
                if st.button("🔊", key=message_key, help="Read aloud", use_container_width=True):
                    with st.spinner("🗣"):
                        speak_text(message["content"], message_key)
            else:
                st.button("🔇", key=f"no_tts_{i}", help="TTS not available", disabled=True, use_container_width=True)
    else:
        # User messages without speaker button
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

# --- INTEGRATED INPUT SECTION ---
st.markdown("---")

# Language selection dropdown
st.markdown("#### 🌍 Select Language / भाषा चुनें / மொழி தேர்வு")
language_options = {
    "English": "English",
    "தமிழ் (Tamil)": "Tamil", 
    "हिंदी (Hindi)": "Hindi"
}

selected_language = st.selectbox(
    "Choose your preferred language:",
    options=list(language_options.keys()),
    index=0,  # Default to English
    key="language_selector"
)

# Get the language value for backend
language_value = language_options[selected_language]

# Get input value (voice text if available, otherwise empty)
input_value = st.session_state.voice_text if st.session_state.voice_text else ""

# Create columns for input area with voice button
col1, col2 = st.columns([6, 1])

with col1:
    # Main text input
    prompt = st.text_area(
        "💭 Type your message:",
        value=input_value,
        height=120,
        placeholder="Type your message here...",
        key="main_text_input"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
    
    if SPEECH_RECOGNITION_AVAILABLE:
        # Voice recording button integrated next to text area
        audio_file = st.audio_input("🎤", key="audio_input", label_visibility="collapsed")
        
        if audio_file is not None:
            # Create a hash to check if this is a new recording
            import hashlib
            audio_content = audio_file.read()
            audio_file.seek(0)  # Reset file pointer
            current_audio_hash = hashlib.md5(audio_content).hexdigest()
            
            # Only process if this is a new recording
            if current_audio_hash != st.session_state.last_audio_hash:
                st.session_state.last_audio_hash = current_audio_hash
                
                # Automatically transcribe when audio is recorded
                with st.spinner("🔄"):
                    transcribed_text = transcribe_audio(audio_file)
                    
                    if not transcribed_text.startswith("❌"):
                        st.session_state.voice_text = transcribed_text
                        st.success("✓")
                        st.rerun()
                    else:
                        st.error("✗")
                        # Show error in a more user-friendly way
                        if "Could not understand" in transcribed_text:
                            st.caption("Speak clearly")
                        elif "internet connection" in transcribed_text:
                            st.caption("Check internet")
                        else:
                            st.caption("Try again")
    else:
        st.info("🎤\nInstall\nSpeechRecognition")

# Send button
col_send1, col_send2, col_send3 = st.columns([2, 3, 2])
with col_send2:
    send_button = st.button("📤 Send Message", use_container_width=True, type="primary")

# Clear voice text after modification
if st.session_state.voice_text and prompt != st.session_state.voice_text:
    st.session_state.voice_text = ""

# Process message
if send_button and prompt.strip():
    # Clear voice state
    st.session_state.voice_text = ""
    st.session_state.audio_processed = False
    st.session_state.last_audio_hash = None  # Reset audio hash
    
    # Add user message
    st.session_state.chat_history[st.session_state.current_chat_key].append(
        {"role": "user", "content": prompt.strip()}
    )
    
    # Display user message
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt.strip())
    
    # Generate and display assistant response
    col1, col2 = st.columns([10, 1])
    
    with col1:
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🤔 Thinking..."):
                has_documents = len(st.session_state.processed_documents) > 0
                chat_history = st.session_state.chat_history[st.session_state.current_chat_key][:-1]
                
                bot_reply = generate_intelligent_reply(
                    prompt.strip(), 
                    has_documents=has_documents,
                    chat_history=chat_history,
                    language=language_value
                )
            
            st.markdown(bot_reply)
    
    with col2:
        if TTS_AVAILABLE:
            # Speaker button for the new response
            new_message_key = f"speak_new_{len(st.session_state.chat_history[st.session_state.current_chat_key])}"
            
            if st.button("🔊", key=new_message_key, help="Read aloud", use_container_width=True):
                with st.spinner("🗣"):
                    speak_text(bot_reply, new_message_key)
        else:
            st.button("🔇", key="no_tts_new", help="TTS not available", disabled=True, use_container_width=True)
    
    # Save assistant response
    st.session_state.chat_history[st.session_state.current_chat_key].append(
        {"role": "assistant", "content": bot_reply}
    )
    
    # Save chat history
    save_chat_history(st.session_state.chat_history)
    st.rerun()

elif send_button and not prompt.strip():
    st.warning("⚠ Please enter a message before sending!")