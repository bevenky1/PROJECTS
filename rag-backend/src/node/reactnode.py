"""LangGraph nodes for RAG workflow + ReAct Agent inside generate_content"""

import uuid
from typing import List, Optional
from src.state.rag_state import RAGState

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# Wikipedia tool
from langchain_community.utilities import WikipediaAPIWrapper
from src.config.logger import get_logger

logger = get_logger(__name__)


class RAGNodes:
    """Contains node functions for RAG workflow"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None  # lazy-init agent

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """Classic retriever node"""
        logger.info("Retrieving documents (reactnode) for question: %s", state.question)
        docs = self.retriever.invoke(state.question)
        logger.debug("Retrieved %d documents", len(docs) if docs else 0)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )

    def _build_tools(self) -> List:
        """Build retriever + wikipedia tools"""
        
        retriever = self.retriever  # Capture in closure
        
        @tool
        def search_documents(query: str) -> str:
            """Fetch passages from indexed corpus based on the query."""
            docs: List[Document] = retriever.invoke(query)
            if not docs:
                return "No documents found."
            merged = []
            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata if hasattr(d, "metadata") else {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")
            return "\n\n".join(merged)

        wiki_wrapper = WikipediaAPIWrapper(top_k_results=3, lang="en")
        
        @tool
        def search_wikipedia(query: str) -> str:
            """Search Wikipedia for general knowledge on a topic."""
            return wiki_wrapper.run(query)

        return [search_documents, search_wikipedia]

    def _build_agent(self):
        """ReAct agent with tools"""
        tools = self._build_tools()
        system_prompt = (
            "You are a helpful RAG agent. "
            "Prefer 'search_documents' for user-provided docs; use 'search_wikipedia' for general knowledge. "
            "Return only the final useful answer."
        )
        self._agent = create_react_agent(self.llm, tools=tools, prompt=system_prompt)
        logger.info("ReAct agent built with %d tools", len(tools))

    def generate_answer(self, state: RAGState) -> RAGState:
        """
        Generate answer using ReAct agent with retriever + wikipedia.
        """
        if self._agent is None:
            logger.debug("Agent not initialized; building agent")
            self._build_agent()

        logger.info("Invoking ReAct agent for question: %s", state.question)
        result = self._agent.invoke({"messages": [HumanMessage(content=state.question)]})

        messages = result.get("messages", [])
        answer: Optional[str] = None
        if messages:
            answer_msg = messages[-1]
            answer = getattr(answer_msg, "content", None)

        logger.debug("Agent produced answer length: %s", len(answer or ""))

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer or "Could not generate answer."
        )
