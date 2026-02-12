
import pytest
import os
import csv

def pytest_sessionfinish(session, exitstatus):
    """
    Called after whole test run finished, right before returning the exit status to the system.
    Generates a detailed CSV report with RAG responses and Judge scores if results exist.
    """
    if hasattr(pytest, "rag_results"):
        report_path = os.path.join(session.config.rootdir, "tests", "detailed_rag_report.csv")
        fieldnames = ["question", "response", "sources", "score", "reasoning"]
        
        with open(report_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for res in pytest.rag_results:
                # Convert list of sources to a string
                sources_str = ", ".join(res["sources"])
                writer.writerow({
                    "question": res["question"],
                    "response": res["response"],
                    "sources": sources_str,
                    "score": res["score"],
                    "reasoning": res["reasoning"]
                })
        
        print(f"\nDetailed RAG report generated at: {report_path}")
