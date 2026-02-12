
import os
import pytest
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from src.rag.engine import RAGEngine
from src.rag.vector_store import initialize_vector_store
from config.prompts import TEST_QUESTIONS
from langchain_ollama import OllamaLLM
from src.rag.embeddings import get_embedding_function

# RAGAS LLM/Embedding Wrappers
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Ground truth mapping for key questions
GROUND_TRUTH = {
    "What is Air India's baggage policy for international flights?": "Most international flights allow 2 pieces of 23kg each for Economy Class and 2 pieces of 32kg each for Business Class.",
    "Tell me about Air India's history and its founders.": "Air India was founded by J.R.D. Tata in 1932 as Tata Airlines. It was nationalized in 1953 and later re-acquired by the Tata Group in 2022.",
    "What are the major air disasters associated with Air India in its history?": "Notable disasters include the 1950 Mont Blanc crash and the 1985 Kanishka bombing (Flight 182).",
    "List some of the domestic routes operated by Air India as of February 2025.": "Air India operates a wide network connecting major cities like Delhi, Mumbai, Bengaluru, Kolkata, and Chennai.",
    "What are the key service regulations for AIESL employees mentioned in the documents?": "AIESL employees are governed by specific service regulations covering working hours, safety protocols, and disciplinary procedures as detailed in the service manual.",
    "Who are the major interline partners or associated airlines for Air India?": "Air India is a member of the Star Alliance and has numerous interline partners including Lufthansa, Singapore Airlines, and United Airlines."
}

@pytest.fixture(scope="module")
def rag_engine():
    """Initializes the RAG engine for evaluation."""
    vs_manager = initialize_vector_store()
    return RAGEngine(vs_manager)

def test_ragas_metrics(rag_engine):
    """
    Evaluates the RAG pipeline using the RAGAS framework.
    Calculates Faithfulness, Answer Relevancy, Context Precision, and Context Recall.
    """
    # 1. Prepare the evaluation dataset
    eval_data = []
    
    # We'll use a subset of questions to avoid overloading memory
    test_subset = [q for q in TEST_QUESTIONS if q in GROUND_TRUTH]
    
    print(f"\nStarting RAGAS evaluation on {len(test_subset)} questions...")
    
    for question in test_subset:
        print(f"Generating response for: {question}")
        response, sources = rag_engine.generate_response(question)
        
        # Retrieve context strings for RAGAS
        docs = rag_engine.vector_store.similarity_search(question, k=3)
        contexts = [doc.page_content for doc in docs]
        
        eval_data.append({
            "question": question,
            "answer": response,
            "contexts": contexts,
            "ground_truth": GROUND_TRUTH[question]
        })
    
    dataset = Dataset.from_list(eval_data)
    
    # 2. Configure RAGAS to use local models
    from config.settings import LOCAL_LLM_MODEL
    
    # Initialize the LLM and Embedding objects
    # Note: Using the 1b model as configured in .env for speed/memory
    llm = OllamaLLM(model=LOCAL_LLM_MODEL)
    embeddings = get_embedding_function()
    
    # Wrap them for RAGAS
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_emb = LangchainEmbeddingsWrapper(embeddings)
    
    # 3. Running evaluation
    print("Computing RAGAS metrics (this may take a few minutes)...")
    try:
        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=ragas_llm,
            embeddings=ragas_emb,
            raise_exceptions=False # Don't crash if one metric fails
        )
        
        # 4. Process and Save results
        print("\n--- RAGAS Metric Scores ---")
        for metric, score in result.items():
            print(f"{metric}: {score:.4f}")
            
        df = result.to_pandas()
        report_path = "tests/ragas_report.csv"
        df.to_csv(report_path, index=False)
        print(f"\nDetailed RAGAS report saved to: {report_path}")
        
        # Basic assertions to ensure metrics are being calculated
        assert result["faithfulness"] >= 0
        assert result["answer_relevancy"] >= 0
        
    except Exception as e:
        pytest.fail(f"RAGAS evaluation failed: {e}")

if __name__ == "__main__":
    # If run directly, initialize engine and run the test
    # (Mocking pytest for absolute local run)
    vs_manager = initialize_vector_store()
    engine = RAGEngine(vs_manager)
    test_ragas_metrics(engine)
