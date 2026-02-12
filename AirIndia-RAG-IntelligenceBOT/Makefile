
.PHONY: install run ingest clean format lint report

PYTHON = python3
PIP = pip

install:
	$(PIP) install -r requirements.txt

run:
	streamlit run app.py

ingest:
	$(PYTHON) src/ingest_data.py --docs-dir AirIndia

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf src/__pycache__
	rm -rf config/__pycache__
	rm -rf tests/__pycache__
	rm -rf logs/*.log

format:
	black .
	isort .

lint:
	flake8 .

report:
	pytest -v tests/test_rag_quality.py --csv=tests/report.csv

ragas-report:
	pytest -v tests/test_ragas_eval.py -s

combined-report:
	pytest -v tests/test_combined_eval.py -s

