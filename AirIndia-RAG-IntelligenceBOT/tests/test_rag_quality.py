
import os
import pytest
from src.rag.engine import RAGEngine
from src.rag.vector_store import initialize_vector_store
from config.prompts import TEST_QUESTIONS

def load_questions():
    """Load questions from the config or sample_questions.txt file."""
    if TEST_QUESTIONS:
        return TEST_QUESTIONS
        
    questions_file = os.path.join(os.path.dirname(__file__), "sample_questions.txt")
    if not os.path.exists(questions_file):
        return ["What is Air India's baggage policy?"]
    
    with open(questions_file, "r") as f:
        return [line.strip() for line in f if line.strip()]

# Load questions at collection time
QUESTIONS = load_questions()

@pytest.fixture(scope="module")
def rag_engine():
    """Fixture to initialize the RAG Engine once for all tests in this module."""
    vs_manager = initialize_vector_store()
    return RAGEngine(vs_manager)

# Initialize a global history for the test session
TEST_SESSION_HISTORY = []

def test_vector_store_not_empty(rag_engine):
    """Verify that the vector store actually contains some documents."""
    # We'll do a broad search to see if anything comes back
    docs = rag_engine.vector_store.similarity_search("Air India", k=1)
    assert len(docs) > 0, "Vector store appears to be empty. Did you run ingestion?"

@pytest.mark.parametrize("question", QUESTIONS)
def test_rag_response_quality(rag_engine, question):
    """Test the RAG engine's response with session-level chat history persistence."""
    global TEST_SESSION_HISTORY
    
    # 1. Generate core RAG response using current test session history
    response, sources = rag_engine.generate_response(question, chat_history=TEST_SESSION_HISTORY)
    
    # 2. Update session history for the NEXT test in the list
    TEST_SESSION_HISTORY.append({"role": "user", "content": question})
    TEST_SESSION_HISTORY.append({"role": "assistant", "content": response})

    # 3. Retrieve the context used for transparency in evaluation
    # (Using the same search query logic as the engine would internally)
    search_query = question
    if len(TEST_SESSION_HISTORY) > 2: # If it's not the first question
        # We don't need to re-run the condense prompt here, we'll just check if it's a meta question
        is_meta = any(word in question.lower() for word in ["asked you", "previous questions", "our conversation", "my last question"])
        if is_meta:
            context = "Conversation History"
        else:
            docs = rag_engine.vector_store.similarity_search(question)
            context = "\n\n".join([doc.page_content for doc in docs])
    else:
        docs = rag_engine.vector_store.similarity_search(question)
        context = "\n\n".join([doc.page_content for doc in docs])

    # 4. Perform LLM-based Evaluation (Judge)
    eval_result = rag_engine.evaluate_response(question, response, context)
    score = eval_result.get("score", 0)
    reasoning = eval_result.get("reasoning", "No reasoning provided")

    # Store results in metadata for report inclusion
    pytest.rag_results = getattr(pytest, "rag_results", [])
    pytest.rag_results.append({
        "question": question,
        "response": response,
        "sources": sources,
        "score": score,
        "reasoning": reasoning
    })

    # Assertions for basic quality
    assert response is not None, f"Response for '{question}' was None"
    assert len(response) > 20, f"Response for '{question}' was too short"
    
    # Assertions for Judge Score (Require at least 3 out of 5)
    assert score >= 3, f"LLM Judge gave a low score ({score}/5) for '{question}'. Reasoning: {reasoning}"
    
    # Assert that sources are found
    assert len(sources) > 0, f"No sources retrieved for question: {question}"

    # Print for visibility
    print(f"\n--- Evaluation for: {question} ---")
    print(f"Response: {response}")
    print(f"Sources: {sources}")
    print(f"Judge Score: {score}/5")
    print(f"Judge Reasoning: {reasoning}")
    print("-" * 50)
