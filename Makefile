.PHONY: install test lint bench all

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

bench:
	python benchmarks/bench.py --write

all: lint test bench
