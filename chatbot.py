import asyncio
import os

import streamlit as st
from agents import Agent, Runner, function_tool
from duckduckgo_search import DDGS
from dotenv import load_dotenv
from agents.mcp import MCPServerSse

# Load environment variables from .env file
load_dotenv()


def check_openai_api_key() -> None:
    """Check if OpenAI API key is available in environment."""
    api_key = os.getenv("SECRET")
    if not api_key:
        st.error("SECRET not found! Please set it in your .env file or environment.")
        st.stop()


@function_tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo.
    
    Args:
        query: The search query to execute.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            if not results:
                return f"No search results found for: {query}"
            
            search_summary = f"Search results for '{query}':\n\n"
            for i, result in enumerate(results, 1):
                search_summary += f"{i}. **{result['title']}**\n"
                search_summary += f"   {result['body']}\n"
                search_summary += f"   Source: {result['href']}\n\n"
            
            return search_summary
    except Exception as e:
        return f"Error performing web search: {e!s}"


async def get_agent_response(messages: list[dict[str, str]]) -> str:
    """Get response from OpenAI agent with web search capability."""
    try:
        # Use MCP server within async context manager
        async with MCPServerSse(
            name="Postgres MCP Server",
            params={
                "url": "http://localhost:8000/sse",
            },
        ) as mcp_server:
            # Create agent with web search tool and MCP server
            agent = Agent(
                name="Web Search Assistant",
                instructions="""You are a helpful assistant with web search capabilities and access to a PostgreSQL database. 
                When users ask questions that require current information or specific facts, 
                use the web_search tool to find relevant information before responding.
                When users ask about database or data-related questions, you can use the database tools.
                Always cite your sources when using search results.""",
                model="gpt-4o-mini",
                tools=[web_search],
                mcp_servers=[mcp_server]
            )
            
            # Convert message history to input string
            user_messages = [msg["content"] for msg in messages if msg["role"] == "user"]
            latest_query = user_messages[-1] if user_messages else ""
            
            # Run the agent
            result = await Runner.run(starting_agent=agent, input=latest_query)
            return result.final_output or ""
        
    except Exception as e:
        st.error(f"Error calling OpenAI Agent: {e!s}")
        return ""


def main() -> None:
    """Main Streamlit chatbot application."""
    st.title("🤖 AI Chatbot")
    st.write("Chat with OpenAI's GPT model!")

    # Check OpenAI API key
    check_openai_api_key()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]

    # Display chat messages
    for message in st.session_state.messages[1:]:  # Skip system message
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"), st.spinner("Searching and thinking..."):
            response = asyncio.run(get_agent_response(st.session_state.messages))
            st.markdown(response)

        # Add assistant response to chat history
        if response:
            st.session_state.messages.append({"role": "assistant", "content": response})

    # Sidebar with controls
    with st.sidebar:
        st.header("Chat Controls")
        if st.button("Clear Chat History"):
            st.session_state.messages = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]
            st.rerun()

        st.markdown("---")
        st.markdown("### Instructions")
        st.markdown("1. Create a `.env` file with `OPENAI_API_KEY=your_key_here`")
        st.markdown("2. Type your message in the chat input")
        st.markdown("3. Press Enter to send")
        st.markdown("4. The assistant can search the web for current information!")
        st.markdown("5. Ask about the movie database - it has 1000+ films, customers, rentals!")
        
        st.markdown("---")
        st.markdown("### Features")
        st.markdown("✅ **Web Search**: Ask about current events, weather, news")
        st.markdown("✅ **Database Access**: Query PostgreSQL database with 1000+ films")
        st.markdown("✅ **Intelligent Routing**: Automatically searches when needed")
        st.markdown("✅ **Source Citations**: References included in responses")


if __name__ == "__main__":
    main()