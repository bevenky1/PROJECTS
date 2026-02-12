
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
from config.settings import LOCAL_LLM_MODEL

# Initialize a global list to store results from all tests
COMBINED_RESULTS = []

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

@pytest.mark.parametrize("question", [q for q in TEST_QUESTIONS if q in GROUND_TRUTH])
def test_collect_data(rag_engine, question):
    """
    Step 1: Run RAG generation and LLM-as-a-Judge for each question.
    Store the data for Step 2 (RAGAS batch evaluation).
    """
    print(f"\nProcessing: {question}")
    
    # 1. RAG Generation
    response, sources = rag_engine.generate_response(question)
    
    # 2. Retrieve Context
    docs = rag_engine.vector_store.similarity_search(question, k=3)
    contexts = [doc.page_content for doc in docs]
    context_text = "\n\n".join(contexts)

    # 3. LLM-as-a-Judge Evaluation (Internal)
    judge_result = rag_engine.evaluate_response(question, response, context_text)
    judge_score = judge_result.get("score", 0)
    judge_reasoning = judge_result.get("reasoning", "No reasoning provided")

    # 4. Store Data for RAGAS
    COMBINED_RESULTS.append({
        "question": question,
        "answer": response,
        "contexts": contexts,
        "ground_truth": GROUND_TRUTH[question],
        "judge_score": judge_score,
        "judge_reasoning": judge_reasoning,
        "sources": ", ".join(sources)
    })

def test_run_ragas_and_save_report():
    """
    Step 2: Run RAGAS evaluation on the collected data and save the combined report.
    This runs ONCE after all questions are processed.
    """
    if not COMBINED_RESULTS:
        pytest.skip("No data collected for RAGAS evaluation.")
        
    print(f"\nRunning RAGAS evaluation on {len(COMBINED_RESULTS)} items...")
    
    # 1. Prepare Dataset
    # We only keep the columns RAGAS needs for evaluation
    ragas_data = {
        "question": [x["question"] for x in COMBINED_RESULTS],
        "answer": [x["answer"] for x in COMBINED_RESULTS],
        "contexts": [x["contexts"] for x in COMBINED_RESULTS],
        "ground_truth": [x["ground_truth"] for x in COMBINED_RESULTS]
    }
    dataset = Dataset.from_dict(ragas_data)
    
    # 2. Configure RAGAS
    # Initialize the LLM and Embedding objects
    # Note: Using the 1b model as configured in .env for speed/memory
    llm = OllamaLLM(model=LOCAL_LLM_MODEL)
    embeddings = get_embedding_function()
    
    # Wrap them for RAGAS
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_emb = LangchainEmbeddingsWrapper(embeddings)
    
    # 3. Predict Metrics
    try:
        results = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=ragas_llm,
            embeddings=ragas_emb,
            raise_exceptions=False
        )
        
        # 4. Merge Results
        df_ragas = results.to_pandas()
        print(f"RAGAS Output Columns: {df_ragas.columns.tolist()}")
        
        # We need to merge this back with our COMBINED_RESULTS (judge scores)
        df_combined = pd.DataFrame(COMBINED_RESULTS)
        
        # Start with our base data
        final_df = df_combined.copy()
        
        # Add RAGAS metrics (concat by index, assuming order is preserved)
        # We only want the metric columns from RAGAS, not the duplicates
        metric_cols = [col for col in df_ragas.columns if col not in ["question", "answer", "contexts", "ground_truth"]]
        
        for col in metric_cols:
            final_df[col] = df_ragas[col]
            
        # Rename internal judge columns to match desired output if needed, or just use what we have
        # COMBINED_RESULTS keys: question, answer, contexts, ground_truth, judge_score, judge_reasoning, sources
        
        # Reorder columns for readability - use actual keys from COMBINED_RESULTS
        cols_order = ["question", "answer", "judge_score", "faithfulness", "answer_relevancy", "context_precision", "context_recall", "judge_reasoning", "sources", "ground_truth", "contexts"]
        
        # Filter for cols that actually exist
        existing_cols = [c for c in cols_order if c in final_df.columns]
        final_df = final_df[existing_cols]

        report_path = "tests/combined_rag_report.csv"
        
        # Rename for cleaner report
        final_df.rename(columns={"judge_score": "llm_judge_score", "judge_reasoning": "llm_judge_reasoning"}, inplace=True)

        report_path = "tests/combined_rag_report.csv"
        final_df.to_csv(report_path, index=False)
        print(f"\nCombined RAG Evaluation Report saved to: {report_path}")
        print(final_df[["question", "llm_judge_score", "faithfulness", "answer_relevancy"]])

    except Exception as e:
        pytest.fail(f"RAGAS evaluation failed: {e}")
