import streamlit as st
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.database import get_db_connection
from src.llm import get_llm
from src.agent import create_agent
from src.logger import logger

st.set_page_config(page_title="Sales Data Chatbot", page_icon="📊")

st.title("📊 Sales Data Chatbot")
st.markdown("Ask questions about your sales data in natural language.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize agent (cached resource to avoid reloading on every rerun)
@st.cache_resource
def load_agent_final():
    try:
        db = get_db_connection()
        llm = get_llm()
        return create_agent(db, llm)
    except Exception as e:
        logger.error(f"Error initializing agent: {e}")
        st.error(f"Failed to initialize agent: {e}")
        return None

agent = load_agent_final()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    logger.info(f"User Query: {prompt}")

    # Generate response
    with st.chat_message("assistant"):
        if agent:
            with st.spinner("Thinking..."):
                try:
                    logger.info("Processing query with agent...")
                    # Invoke the agent
                    response = agent.invoke({"messages": [("user", prompt)]})
                    
                    # Log the full response object for deeper inspection
                    logger.info("--------------------------------------------------")
                    logger.info(f"User Query: {prompt}")
                    
                    # Log intermediate steps (Tool Calls)
                    # The response dictionary often contains 'intermediate_steps' if using return_intermediate_steps=True
                    # Since we are using create_react_agent, the messages list contains the tool calls.
                    messages = response.get("messages", [])
                    for msg in messages:
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            for tool_call in msg.tool_calls:
                                logger.info(f"Tool Call: {tool_call['name']}")
                                logger.info(f"Tool Input: {tool_call['args']}")
                        if hasattr(msg, 'content') and msg.content:
                            # Log content but avoid duplicates if it's the final answer
                            pass

                    ai_message = messages[-1].content
                    logger.info(f"Agent Response: {ai_message}")
                    logger.info("--------------------------------------------------")
                    
                    st.markdown(ai_message)
                    st.session_state.messages.append({"role": "assistant", "content": ai_message})
                except Exception as e:
                    logger.error(f"Error processing query '{prompt}': {e}", exc_info=True)
                    st.error(f"Error processing query: {e}")
        else:
            logger.error("Agent not initialized when query received.")
            st.error("Agent not initialized. Please check logs.")
