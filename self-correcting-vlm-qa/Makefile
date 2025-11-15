.PHONY: help install dev test run-api run-demo run clean

help:
	@echo "Self-Correcting VLM QA with Claude - Available Commands"
	@echo "======================================================="
	@echo "install   - Install dependencies"
	@echo "dev       - Install development dependencies"
	@echo "test      - Run tests"
	@echo "run-api   - Run FastAPI server"
	@echo "run-demo  - Run Streamlit demo"
	@echo "run       - Run both API and demo (one command)"
	@echo "clean     - Clean cache and temporary files"

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements.txt
	pip install pytest pytest-asyncio httpx black flake8 mypy

test:
	pytest tests/ -v

run-api:
	python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-demo:
	streamlit run demo/app.py

run:
	./run_demo.sh

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	rm -rf build/ dist/
