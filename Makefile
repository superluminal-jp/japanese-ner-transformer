# Makefile for Japanese NER Transformer

.PHONY: help install install-dev test test-unit test-integration test-e2e test-coverage clean lint format

# Default target
help:
	@echo "Available commands:"
	@echo "  install       - Install production dependencies"
	@echo "  install-dev   - Install development dependencies"
	@echo "  test          - Run all tests"
	@echo "  test-unit     - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-e2e      - Run end-to-end tests only"
	@echo "  test-coverage - Run tests with coverage report"
	@echo "  lint          - Run code linting"
	@echo "  format        - Format code with black and isort"
	@echo "  clean         - Clean temporary files"
	@echo "  demo          - Run simple NER demo"
	@echo "  batch-example - Run batch analysis example"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

# Testing
test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v -m "unit or not (integration or e2e)"

test-integration:
	pytest tests/integration/ -v -m "integration"

test-e2e:
	pytest tests/e2e/ -v -m "e2e"

test-coverage:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# Code quality
lint:
	flake8 src/ tests/ main.py
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ main.py
	isort src/ tests/ main.py

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf output/

# Examples
demo:
	python main.py --demo

batch-example:
	@echo "Creating sample documents..."
	@mkdir -p sample_docs
	@echo "田中太郎は東京大学の教授です。" > sample_docs/doc1.txt
	@echo "佐藤花子はソニー株式会社で働いています。" > sample_docs/doc2.txt
	@echo "山田次郎は大阪で開催されたAI技術カンファレンスに参加しました。" > sample_docs/doc3.txt
	python main.py sample_docs/ -o sample_output
	@echo "Results saved to sample_output/"
	@echo "Check sample_output/analysis_report.md for the report"