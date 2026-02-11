.PHONY: format lint

format:
	black .

lint:
	ruff check .
