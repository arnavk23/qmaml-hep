.PHONY: install test test-all lint format figures summary

install:        ## editable install with dev and plotting extras
	pip install -e ".[dev,plots]"

test:           ## fast suite (2-4 min)
	python -m pytest -q -m "not slow"

test-all:       ## including end-to-end reproduction and the UCCSD simulator check (~6 min)
	python -m pytest -q

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

format:
	ruff check --fix src tests scripts
	ruff format src tests scripts

summary:        ## rescore results/unified -> results/unified_summary.json
	python -m qmaml_hep.analysis

figures:        ## figures/*.pdf and figures/tables/*.tex from results/
	python scripts/make_figures.py
	python scripts/make_tables.py
