PY := uv run python
.PHONY: all ws1 ws2 ws3 ws4 memo storybook app test lint clean

all: ws1 ws2 ws3 ws4 memo

ws1:            ## Data contract, audit, signals, archetypes
	$(PY) -m bikeops.cli ws1

ws2: ws1        ## Hypothesis tests, supply-cap probe, experiment power
	$(PY) -m bikeops.cli ws2

ws3: ws1        ## Walk-forward forecast, conformal intervals, concurrency, fleet sizing
	$(PY) -m bikeops.cli ws3

ws4: ws2 ws3    ## Demand cube + notes index for the desk
	$(PY) -m bikeops.cli ws4

memo: ws4       ## Decision memo + model card refresh
	$(PY) -m bikeops.cli memo

storybook:      ## Execute the three viewing notebooks
	uv run jupyter nbconvert --to notebook --execute --inplace storybook/*.ipynb

app:            ## Launch the Gradio desk
	$(PY) -m bikeops.desk.ui

test: lint
	uv run pytest -q

lint:
	uv run ruff check bikeops tests

clean:
	rm -rf warehouse/*.duckdb artifacts/figures/* artifacts/tables/* artifacts/models/*
