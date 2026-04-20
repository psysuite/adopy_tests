.PHONY: help install test coverage clean

help:
	@echo "Available commands:"
	@echo "  make install   - Install package in development mode"
	@echo "  make test      - Run tests"
	@echo "  make coverage  - Run tests with coverage and open HTML report"
	@echo "  make clean     - Remove build artifacts"

install:
	venv/bin/pip install -e .

test:
	venv/bin/pytest tests/ -v

coverage:
	venv/bin/pytest tests/ -v
	@echo "\nOpening coverage report..."
	@xdg-open build/htmlcov/index.html 2>/dev/null || open build/htmlcov/index.html 2>/dev/null || echo "Open build/htmlcov/index.html in your browser"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf tests/output/*.xlsx tests/output/*.txt 2>/dev/null || true
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
