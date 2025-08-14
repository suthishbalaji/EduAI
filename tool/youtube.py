import os
import json
from typing import List, Dict, Any
from googleapiclient.discovery import build
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class YouTubeSearchInput(BaseModel):
    query: str = Field(description="The search query for YouTube videos")
    max_results: int = Field(default=5, description="Maximum number of results to return")

class YouTubeSearchTool(BaseTool):
    name: str = "youtube_search"
    description: str = "Search for YouTube videos based on a query and return video recommendations with titles, descriptions, and URLs"
    args_schema: type[BaseModel] = YouTubeSearchInput
    
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = False
    
    def _run(self, query: str, max_results: int = 5) -> str:
        try:
            print(f"DEBUG: YouTube tool called with query: '{query}', max_results: {max_results}")
            
            # Get API key
            youtube_api_key = os.getenv("YOUTUBE_API_KEY")
            if not youtube_api_key:
                return "Error: YOUTUBE_API_KEY not found in environment variables"
            
            print(f"DEBUG: API key found: {youtube_api_key[:10]}...")
            
            # Create YouTube client directly in the method to avoid Pydantic issues
            try:
                youtube = build('youtube', 'v3', developerKey=youtube_api_key)
                print("DEBUG: YouTube client created successfully")
            except Exception as client_error:
                return f"Error creating YouTube client: {str(client_error)}"
            
            # Search for videos
            try:
                search_response = youtube.search().list(
                    q=query,
                    part='id,snippet',
                    maxResults=max_results,
                    type='video',
                    order='relevance'
                ).execute()
                print(f"DEBUG: YouTube API call successful, got {len(search_response.get('items', []))} items")
            except Exception as api_error:
                return f"Error calling YouTube API: {str(api_error)}"
            
            videos = []
            for item in search_response['items']:
                video_id = item['id']['videoId']
                title = item['snippet']['title']
                description = item['snippet']['description']
                channel_title = item['snippet']['channelTitle']
                published_at = item['snippet']['publishedAt']
                thumbnail = item['snippet']['thumbnails']['high']['url']
                
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                
                videos.append({
                    'title': title,
                    'description': description[:200] + "..." if len(description) > 200 else description,
                    'channel': channel_title,
                    'published': published_at,
                    'url': video_url,
                    'thumbnail': thumbnail
                })
            
            # Format the results for the agent
            result = f"Found {len(videos)} YouTube videos for '{query}':\n\n"
            for i, video in enumerate(videos, 1):
                result += f"{i}. **{video['title']}**\n"
                result += f"   Channel: {video['channel']}\n"
                result += f"   Description: {video['description']}\n"
                result += f"   URL: {video['url']}\n"
                result += f"   Published: {video['published'][:10]}\n\n"
            
            print(f"DEBUG: Formatted result length: {len(result)}")
            return result
            
        except Exception as e:
            error_msg = f"Error searching YouTube: {str(e)}"
            print(f"DEBUG: {error_msg}")
            import traceback
            traceback.print_exc()
            return error_msg
    
    async def _arun(self, query: str, max_results: int = 5) -> str:
        return self._run(query, max_results)

class YouTubeRecommendationAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Initialize YouTube search tool
        self.youtube_tool = YouTubeSearchTool()
        
        # Create tools list
        self.tools = [self.youtube_tool]
        
        # Create a simpler, more direct prompt
        self.prompt = PromptTemplate.from_template("""
You are a YouTube video recommendation assistant. You have access to a youtube_search tool that finds real YouTube videos.

To answer the user's question, you MUST:
1. Use the youtube_search tool with the user's query
2. Present the actual results returned by the tool

Available tools: {tools}
Tool names: {tool_names}

Format:
Question: {input}
Thought: I need to search for YouTube videos about this topic
Action: youtube_search  
Action Input: {{"query": "{input}", "max_results": 5}}
Observation: [tool will return real YouTube videos]
Final Answer: [present the videos from the observation]

Question: {input}
{agent_scratchpad}
""")

        # Create the agent with simpler configuration
        self.agent = create_react_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=2,  # Reduced iterations
            early_stopping_method="generate",
            handle_parsing_errors=True
        )
    
    def get_recommendations(self, query: str) -> str:
        try:
            print(f"DEBUG: Starting search for query: {query}")
            
            # First, try direct tool usage to ensure it works
            print("DEBUG: Testing YouTube tool directly...")
            direct_result = self.youtube_tool._run(query, 5)
            
            if "Error searching YouTube" in direct_result:
                return f"❌ YouTube API Error: {direct_result}"
            
            print("DEBUG: Direct tool test successful, trying agent...")
            
            # Try the agent with a simple, direct approach
            try:
                response = self.agent_executor.invoke({
                    "input": query
                })
                
                if response and "output" in response:
                    return response["output"]
                else:
                    print("DEBUG: Agent didn't return expected output, using direct result")
                    return self._format_direct_result(direct_result)
                    
            except Exception as agent_error:
                print(f"DEBUG: Agent failed: {agent_error}, using direct result")
                return self._format_direct_result(direct_result)
                
        except Exception as e:
            print(f"DEBUG: Error in get_recommendations: {str(e)}")
            return f"Error getting recommendations: {str(e)}"
    
    def _format_direct_result(self, result: str) -> str:
        """Format the direct tool result for better presentation"""
        return f"## 🎥 YouTube Video Recommendations\n\n{result}\n\n*Recommendations powered by YouTube Data API and Gemini AI*"

def main():
    st.set_page_config(
        page_title="YouTube Video Recommendation Agent",
        page_icon="🎥",
        layout="wide"
    )
    
    st.title("🎥 YouTube Video Recommendation Agent")
    st.markdown("Enter a topic and get AI-powered YouTube video recommendations!")
    
    # Debug: Show API key status
    youtube_key = os.getenv("YOUTUBE_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    
    with st.expander("🔧 Debug Info"):
        st.write(f"YouTube API Key loaded: {'✅' if youtube_key else '❌'}")
        st.write(f"Google API Key loaded: {'✅' if google_key else '❌'}")
        if youtube_key:
            st.write(f"YouTube Key preview: {youtube_key[:10]}...{youtube_key[-5:]}")
        if google_key:
            st.write(f"Google Key preview: {google_key[:10]}...{google_key[-5:]}")
    
    # Add mode selection
    use_agent = st.checkbox("Use AI Agent (experimental)", value=False, 
                           help="Check this to use the LangChain agent. Uncheck for direct YouTube search.")
    
    # Initialize the agent
    try:
        if 'agent' not in st.session_state:
            with st.spinner("Initializing agent..."):
                st.session_state.agent = YouTubeRecommendationAgent()
                st.success("✅ Agent initialized successfully!")
        
        # User input
        query = st.text_input(
            "What topic would you like YouTube video recommendations for?",
            placeholder="e.g., machine learning tutorials, cooking pasta, guitar lessons"
        )
        
        if st.button("Get Recommendations", type="primary"):
            if query:
                with st.spinner("Searching for video recommendations..."):
                    if use_agent:
                        st.info("🤖 Using AI Agent mode...")
                        recommendations = st.session_state.agent.get_recommendations(query)
                    else:
                        st.info("🔍 Using Direct Search mode...")
                        recommendations = st.session_state.agent.get_recommendations(query)
                    
                    st.markdown("## 📺 Recommended Videos")
                    st.markdown(recommendations)
            else:
                st.warning("Please enter a topic to search for!")
        
        # Display usage instructions
        with st.expander("ℹ️ How to use"):
            st.markdown("""
            1. Enter any topic you're interested in learning about
            2. Click "Get Recommendations" 
            3. The AI agent will search YouTube and provide personalized recommendations
            4. Each recommendation includes title, description, channel, and direct link
            
            **Example queries:**
            - "Python programming for beginners"
            - "How to bake chocolate cake"
            - "Photography composition techniques"
            - "Yoga for back pain relief"
            """)
            
    except Exception as e:
        st.error(f"Error initializing the agent: {str(e)}")
        st.markdown("""
        **Setup Instructions:**
        1. Make sure you have set up your API keys in the `.env` file
        2. You need both YOUTUBE_API_KEY and GOOGLE_API_KEY (for Gemini)
        3. Install all required packages: `pip install -r requirements.txt`
        4. Restart the application
        """)

if __name__ == "__main__":
    main()