# Fleet or Flow?

A decision-first analysis of the Capital Bikeshare 2011–2012 data (`hour.csv`, `day.csv`) for the CEO's question
*"should we buy more bikes?"*. It profiles and repairs the data, tests what suppresses demand, forecasts demand
probabilistically, simulates how many bikes are actually on the road, sizes the fleet, and gives operations a
question-answering desk over a pre-computed demand cube.

**Start here:** [`docs/decision_memo.md`](docs/decision_memo.md) (two pages for the CEO).

| Task | Deliverable |
|---|---|
| Task 1 — quality checks & features | [`docs/data_audit.md`](docs/data_audit.md), `bikeops/contract.py`, `audit.py`, `signals.py`, `archetypes.py` |
| Task 2 — hypothesis | [`docs/hypothesis_brief.md`](docs/hypothesis_brief.md) (+ [pre-registration](docs/hypothesis_preregistration.md)), `bikeops/inference.py` |
| Task 3 — demand model & fleet | [`docs/model_card.md`](docs/model_card.md), `bikeops/forecast.py`, `concurrency.py`, `fleet.py` |
| Task 4 — natural-language prototype | `bikeops/desk/` (Gradio app), `cube.py`, `tests/test_desk_acceptance.py` |
| Decision | [`docs/decision_memo.md`](docs/decision_memo.md), [`docs/decision_log.md`](docs/decision_log.md) |

## Runbook

```bash
uv sync                      # install dependencies (Python 3.11)
cp .env.example .env         # optional: add ANTHROPIC_API_KEY (or GEMINI_API_KEY)
ollama pull qwen2.5:7b-instruct   # optional offline model (name in config.yaml)
make all                     # full pipeline: raw data -> warehouse -> artifacts -> docs (≈ 30–60 min)
make app                     # launch the Gradio desk (http://127.0.0.1:7860)
make test                    # ruff + pytest
```

Individual stages: `make ws1`, `make ws2`, `make ws3`, `make ws4`, `make memo`. `make storybook` executes the three
viewing notebooks. All seeds, dates, assumptions, costs and LLM provider/model names live in `config.yaml`; keys live in `.env`.

The desk works **without any API key**: `desk.provider: rules` (or no key set) uses a deterministic router and SQL templates.
With a key, LiteLLM routes the question, writes guarded SQL (sqlglot whitelist, SELECT-only, ≤ 500 rows, read-only DuckDB) and
writes the answer from the returned rows only.

## Layout

```
fleet-or-flow/
├── Makefile  pyproject.toml  uv.lock  config.yaml  .env.example
├── raw/                      hour.csv, day.csv, Readme.txt (read-only)
├── warehouse/bikes.duckdb    cleaned tables, signals, backtests, concurrency, demand cube, notes (FTS)
├── bikeops/                  importable package (contract, audit, signals, archetypes, explore, inference,
│   │                         forecast, concurrency, fleet, cube, reporting, cli)
│   └── desk/                 router, sql_agent, tools, resolver, llm, core, ui (Gradio)
├── artifacts/figures|tables|models
├── docs/                     data_audit, hypothesis_brief, model_card, decision_memo, decision_log
├── storybook/                3 thin notebooks that only display artifacts
└── tests/                    pytest suite incl. desk acceptance + guardrails + router set
```

Data: Capital Bikeshare system data with weather, as distributed with the UCI Bike Sharing Dataset (see `raw/Readme.txt`).
