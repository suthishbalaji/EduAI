import os
import requests
import time
import re
from typing import Optional

# Core imports
from langchain.tools import tool
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory

# Third-party imports
from bs4 import BeautifulSoup
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS
from langchain_google_genai import ChatGoogleGenerativeAI

# Environment setup
from dotenv import load_dotenv
load_dotenv()

class Config:
    """Configuration class for the agent"""
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    MAX_SEARCH_RESULTS = 5
    MAX_SCRAPE_LENGTH = 5000
    REQUEST_TIMEOUT = 10
    GEMINI_MODEL = "gemini-1.5-flash"
    TEMPERATURE = 0.7
    USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

# Web Scraping Tool using @tool decorator
@tool
def web_scraper(url: str) -> str:
    """Scrapes content from a given URL and returns clean text.
    
    Args:
        url: URL to scrape content from
        
    Returns:
        Cleaned text content from the URL
    """
    try:
        headers = {
            'User-Agent': Config.USER_AGENT
        }
        
        response = requests.get(url, headers=headers, timeout=Config.REQUEST_TIMEOUT)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Limit text length
        if len(text) > Config.MAX_SCRAPE_LENGTH:
            text = text[:Config.MAX_SCRAPE_LENGTH] + "..."
        
        return f"Content from {url}:\n{text}"
        
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"

# Search Tool using @tool decorator
@tool
def web_search(query: str) -> str:
    """Searches the web for relevant URLs based on a query.
    
    Args:
        query: Search query to find relevant URLs
        
    Returns:
        Formatted search results with URLs and descriptions
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=Config.MAX_SEARCH_RESULTS))
            
            if not results:
                return f"No search results found for: {query}"
            
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append(
                    f"{i}. Title: {result.get('title', 'No title')}\n"
                    f"   URL: {result.get('href', 'No URL')}\n"
                    f"   Snippet: {result.get('body', 'No description')}\n"
                )
            
            return f"Search results for '{query}':\n\n" + "\n".join(formatted_results)
            
    except Exception as e:
        return f"Search error for '{query}': {str(e)}"

# Simple Cache Implementation
class SimpleCache:
    def __init__(self, cache_duration_minutes=30):
        self.cache = {}
        self.cache_duration_minutes = cache_duration_minutes
    
    def _get_key(self, data):
        import hashlib
        return hashlib.md5(str(data).encode()).hexdigest()
    
    def get(self, key):
        cache_key = self._get_key(key)
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_duration_minutes * 60:
                return data
            else:
                del self.cache[cache_key]
        return None
    
    def set(self, key, data):
        cache_key = self._get_key(key)
        self.cache[cache_key] = (data, time.time())

# Main Web Scraping Agent Class
class WebScrapingAgent:
    def __init__(self):
        """Initializes the agent with tools, LLM, and memory."""
        if not Config.GOOGLE_API_KEY:
            raise ValueError("Google API key not found. Please set it in the .env file.")
        
        self.llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            google_api_key=Config.GOOGLE_API_KEY,
            temperature=Config.TEMPERATURE,
            convert_system_message_to_human=True
        )
        
        self.tools = [web_search, web_scraper]
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True
        )
        self.cache = SimpleCache()
        
        self.agent = initialize_agent(
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            llm=self.llm,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
    
    def process_query(self, query: str) -> str:
        """Process user query with web scraping and AI response"""
        try:
            # Check cache first
            cached_result = self.cache.get(query)
            if cached_result:
                return f"[Cached Result] {cached_result}"
            
            # Enhanced prompt for better web scraping behavior
            enhanced_prompt = f"""
            User Query: {query}
            
            Please follow these steps to provide a comprehensive answer:
            
            1. First, use the web_search tool with specific, detailed search terms related to the user's question (try multiple relevant searches if needed)
            2. From the search results, identify the most relevant and reliable URLs that contain actual content about the topic
            3. Use the web_scraper tool to extract detailed content from these URLs
            4. Synthesize the scraped information to provide a comprehensive, practical answer
            5. Focus on giving direct, actionable information rather than just listing sources
            6. Keep your final answer concise but informative (around 8-12 lines unless asked otherwise)
            7. Don't just say the search results aren't relevant - try different search terms or provide general knowledge if needed
            
            Provide a helpful, direct answer based on the most current and relevant information available.
            """
            
            response = self.agent.invoke({"input": enhanced_prompt})
            
            # Cache the result
            response_text = response.get("output", str(response))
            self.cache.set(query, response_text)
            
            return response_text
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(f"DEBUG: {error_msg}")
            return error_msg
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        print("✅ Conversation memory cleared.")
    
    def get_memory_stats(self):
        """Get memory usage statistics"""
        try:
            buffer = self.memory.chat_memory.messages
            return f"Memory contains {len(buffer)} messages."
        except:
            return "Memory statistics not available."

# Utility Functions
def validate_api_key():
    """Validate if the Google API key is set"""
    if not Config.GOOGLE_API_KEY:
        print("❌ Error: GOOGLE_API_KEY not found!")
        print("\n📋 Setup Instructions:")
        print("1. Go to https://makersuite.google.com/app/apikey")
        print("2. Create a new API key")
        print("3. Create a .env file with: GOOGLE_API_KEY=your_api_key_here")
        print("4. Or set environment variable: export GOOGLE_API_KEY=your_api_key_here")
        return False
    return True

def print_welcome():
    """Print welcome message"""
    print("=" * 60)
    print("🤖 LangChain Web Scraping Agent with Gemini API")
    print("=" * 60)
    print("\n🌟 Features:")
    print("  • Real-time web search and scraping")
    print("  • AI-powered response generation")
    print("  • Conversation memory")
    print("  • Source citation")
    print("  • Intelligent content synthesis")
    print("\n💡 Commands:")
    print("  • Type your question and press Enter")
    print("  • 'clear' - Clear conversation memory")
    print("  • 'stats' - Show memory statistics")
    print("  • 'quit' - Exit the program")
    print("\n" + "=" * 60)

def main():
    """Main function to run the agent"""
    
    # Validate API key
    if not validate_api_key():
        return
    
    # Print welcome message
    print_welcome()
    
    # Initialize agent
    try:
        print("\n🔄 Initializing agent...")
        agent = WebScrapingAgent()
        print("✅ Agent initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return
    
    # Main interaction loop
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print("👋 Thank you for using the Web Scraping Agent. Goodbye!")
                break
            
            elif user_input.lower() == 'clear':
                agent.clear_memory()
                continue
            
            elif user_input.lower() == 'stats':
                stats = agent.get_memory_stats()
                print(f"📊 {stats}")
                continue
            
            elif user_input.lower() == 'help':
                print("\n🆘 Available Commands:")
                print("  • Ask any question for web research")
                print("  • 'clear' - Clear conversation memory")
                print("  • 'stats' - Show memory statistics")
                print("  • 'help' - Show this help message")
                print("  • 'quit' - Exit the program")
                continue
            
            # Skip empty input
            if not user_input:
                print("❓ Please enter a question or command.")
                continue
            
            # Process query
            print("\n🔍 Searching the web and generating response...")
            print("-" * 50)
            
            response = agent.process_query(user_input)
            
            print(f"\n🤖 Agent Response:")
            print("-" * 50)
            print(response)
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n\n👋 Program interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try again or type 'quit' to exit.")

# Sample usage and testing
def run_sample_queries():
    """Run some sample queries for testing"""
    if not validate_api_key():
        return
    
    sample_queries = [
        "What are the latest AI developments in 2024?",
        "Current weather in New York",
        "Recent news about electric vehicles",
        "Best Python libraries for web scraping"
    ]
    
    print("🧪 Running sample queries for testing...\n")
    
    try:
        agent = WebScrapingAgent()
        
        for i, query in enumerate(sample_queries, 1):
            print(f"\n{'='*50}")
            print(f"Sample Query {i}: {query}")
            print('='*50)
            
            response = agent.process_query(query)
            print(f"\nResponse: {response}")
            
            time.sleep(2)  # Delay between queries
            
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    # Check if we want to run in test mode
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_sample_queries()
    else:
        main()

"""
SETUP INSTRUCTIONS:
==================

1. Install required packages:
   pip install langchain langchain-google-genai langchain-community beautifulsoup4 requests python-dotenv duckduckgo-search lxml

2. Create a .env file with your Gemini API key:
   GOOGLE_API_KEY=your_gemini_api_key_here

3. Get your free Gemini API key:
   - Go to https://makersuite.google.com/app/apikey
   - Create a new API key
   - Copy it to your .env file

4. Run the script:
   python agent.py

5. For testing mode:
   python agent.py --test

FEATURES:
=========
✅ Real-time web search using DuckDuckGo
✅ Intelligent web scraping with content cleaning
✅ AI-powered response generation using Gemini
✅ Conversation memory for context
✅ Response caching to improve performance
✅ Error handling and robust error recovery
✅ Source citation in responses
✅ Interactive command-line interface
✅ Memory management commands
✅ Sample query testing mode

EXAMPLE QUERIES:
===============
- "What are the latest developments in artificial intelligence?"
- "Current stock price of Apple"
- "Recent news about climate change"
- "Best practices for Python programming in 2024"
- "Latest updates on space exploration"

The agent will automatically search the web, scrape relevant content, 
and provide intelligent responses with proper source citations.
"""