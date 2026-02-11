.PHONY: format lint test

format:
	black .

lint:
	ruff check .

test:
	pytest
