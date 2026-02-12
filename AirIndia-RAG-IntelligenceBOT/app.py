
import streamlit as st
import time
from src.rag.engine import RAGEngine
from src.rag.vector_store import initialize_vector_store
from src.logger import setup_logger
from config.settings import MODEL_TYPE, LOCAL_LLM_MODEL, BEDROCK_MODEL_ID, LOG_FILE
from config.prompts import UI_SAMPLE_QUESTIONS, CHAT_INPUT_PLACEHOLDER
import os

logger = setup_logger(__name__)

# Initialize application components
try:
    vector_store_manager = initialize_vector_store()
    rag_engine = RAGEngine(vector_store_manager)
except Exception as e:
    st.error(f"Failed to initialize System: {e}")
    st.stop()

st.set_page_config(
    page_title="AirIndia-RAG-IntelligenceBOT",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for additional info / controls
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=200)
    else:
        st.title("✈️ RAG-IntelligenceBOT")
    st.title("About")
    st.markdown(f"""
    This intelligent assistant helps you navigate Air India's policies, operations, and more.
    
    It uses advanced RAG (Retrieval-Augmented Generation) technology powered by **{MODEL_TYPE.upper()}** and **ChromaDB**.
    """)
    
    active_model = BEDROCK_MODEL_ID if MODEL_TYPE == "bedrock" else LOCAL_LLM_MODEL
    st.metric(label="Model Mode", value=MODEL_TYPE.upper())
    st.metric(label="Active Model", value=active_model)
    st.divider()

    st.subheader("📜 System Logs")
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            logs = f.readlines()
            # Show last 20 lines
            log_text = "".join(logs[-20:])
            st.code(log_text, language="text")
    else:
        st.info("No logs found yet.")

    st.divider()
    st.caption("© 2024 AirIndia-RAG-IntelligenceBOT Project")

# Main Interface
st.title("✈️ AirIndia-RAG-IntelligenceBOT")
st.markdown("ask me anything about Air India's services, baggage rules, or flight operations.")

# Sample Questions Section
st.subheader("💡 Sample Questions")
cols = st.columns(2)
samples = UI_SAMPLE_QUESTIONS

for i, q in enumerate(samples):
    if cols[i % 2].button(q, use_container_width=True):
        st.session_state.sample_prompt = q


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.write(f"- {source}")

# React to user input
input_val = st.session_state.get('sample_prompt', None)
if prompt := st.chat_input(CHAT_INPUT_PLACEHOLDER):
    pass
elif input_val:
    prompt = input_val
    del st.session_state.sample_prompt

if prompt:
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Analyzing documents..."):
            try:
                # Log the user query for analytics/debugging
                logger.info(f"Processing user query: {prompt}")
                
                # Pass the history (everything except the current message just added)
                response, sources = rag_engine.generate_response(prompt, chat_history=st.session_state.messages[:-1])
                
                # Simulate typing effect
                for chunk in response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                
                if sources:
                    with st.expander("📚 Sources"):
                        for source in sources:
                            st.write(f"- {source}")
            except Exception as e:
                logger.error(f"Error processing query '{prompt}': {e}", exc_info=True)
                st.error(f"An error occurred: {e}")
                full_response = "I'm sorry, I couldn't process your request at the moment."
                sources = []
                message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response,
        "sources": sources
    })
