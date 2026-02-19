PYTHON ?= python

.PHONY: test fetch build backtest regimes sweep stress report

test:
	$(PYTHON) -m pytest

fetch:
	$(PYTHON) -m src.cli fetch --config configs/default.yaml

build:
	$(PYTHON) -m src.cli build --config configs/default.yaml

backtest:
	$(PYTHON) -m src.cli backtest --config configs/default.yaml

regimes:
	$(PYTHON) -m src.cli regimes --config configs/default.yaml

sweep:
	$(PYTHON) -m src.cli sweep --config configs/default.yaml

stress:
	$(PYTHON) -m src.cli stress --config configs/default.yaml

report:
	$(PYTHON) -m src.cli report --config configs/default.yaml
