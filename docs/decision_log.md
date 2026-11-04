# Decision log

Every analytical choice, with the options considered, the choice and the reason. Results that a choice produced are
in the generated docs; this log records *why*.

| Date | Area | Options considered | Choice | Reason |
|---|---|---|---|---|
| 2026-11-04 | Environment | pip/venv · conda · uv | uv + pyproject, Python 3.11 | Locked, fast, reproducible (`uv.lock`); blueprint Part 3 |
| 2026-11-04 | Orchestration | notebooks · Makefile + package | Makefile targets calling `python -m bikeops.cli` | One command rebuilds raw → memo; notebooks only display artifacts |
| 2026-11-04 | Profile facts | trust blueprint · recompute | Recompute all 21 facts in code (`profile_facts.csv`), stop on any mismatch | Part 7 rule; all reproduced |
| 2026-11-04 | Season label | Readme label · meteorological season | `met_season` from month | season 1 = 21 Dec–20 Mar (astronomical), not "spring" |
| 2026-11-04 | Calendar gaps | treat all as zero · drop all · classify | Night lull (00–05, run ≤ 2 h) → 0 rentals, flagged; others = outage, excluded | Short night gaps are almost surely zero-rental hours; daytime/long gaps are system outages (e.g. Sandy) where zero would be false |
| 2026-11-04 | Sentinel zeros | keep · drop rows · impute | Null → same-day interpolation → month×hour median, with flags | 0 % humidity / exact 0 km/h hourly means are recording artefacts; dropping rows would lose rental data |
| 2026-11-04 | weathersit 4 | keep · fold into 3 | Fold into 3 | 3 rows cannot support their own effect |
| 2026-11-04 | Redundancy | temp · atemp · both · PCA | temp only | corr ≈ 0.99; temp is the measured quantity, atemp is derived |
| 2026-11-04 | Anomaly detection | raw z · STL remainder (level) · STL remainder (log) | Robust z (median/MAD) of the (non-robust) STL remainder on log daily totals, \|z\| > 3.5 → 35 days | robust STL flagged 151 days (every rainy day) — not useful; log makes the remainder scale-free as demand grows; robust z is not masked by the anomalies themselves |
| 2026-11-04 | Trend measure | linear fit on levels · STL trend | Slope of log STL trend → % per month | Interpretable growth rate |
| 2026-11-04 | Archetypes | raw hourly counts · hourly shares; KMeans · hierarchical | KMeans on 24-h shares, k by silhouette (2–5), 20 inits, fixed seed | Shape, not level, distinguishes commute vs leisure days |
| 2026-11-04 | Archetypes in forecasting | fit once on all days · refit per fold | Refit per fold on training days; k fixed from WS1 | No look-ahead into test-day shapes; k is a structural choice |
| 2026-11-04 | Partial days | drop · assign on observed hours | Nearest centroid on observed hours (renormalised) | Days with an outage still get an archetype |
| 2026-11-04 | Bridge day | none · workday between holiday and weekend | Workday whose neighbours are both non-working and at least one is a holiday | Captures the Friday after Thanksgiving and Christmas Eve Mondays |
| 2026-11-04 | Comfort index | THI · apparent temp · simple penalty | temp − 0.15·(hum−70)+ − 0.25·(wind−25)+ | Transparent, one number, coefficients in config |
| 2026-11-04 | Daylight | monthly table · astral | astral sunrise/sunset for D.C., hour midpoint rule | Exact by date, deterministic |
| 2026-11-04 | Weather feature | wet only · weathersit category too | wet + `weather_cat` (1–3) as categorical | Mist carries information; it is a raw input, not a target transform |
| 2026-11-04 | H1 test statistic | difference in % · ratio of rate-ratios | RRR (casual RR / registered RR) | Multiplicative, scale-free; natural in a log-link count model |
| 2026-11-04 | Count model | Poisson · NB (discrete MLE) · NB GLM with moment α | NB GLM, α from Cameron–Trivedi auxiliary regression, SEs clustered by date | Overdispersed counts; 200+ fixed effects make full MLE slow; clustering handles within-day dependence |
| 2026-11-04 | Matched design | exact same-day · nearest dates | Each wet hour vs the 2 nearest dry hours, same hour, day class, month, year; bootstrap over days | Model-free check; bootstrap over days respects clustering |
| 2026-11-04 | Pre-registration | write rule in doc later · write file before estimation | `preregister()` writes the rule file first; kept on rebuilds | Part 2.1 d — rule fixed before results |
| 2026-11-04 | Supply-cap probe | skip · residual-variance check | Monthly p95/max + residual CV by predicted-demand quintile; labelled exploratory | Censoring cannot be proven from served demand; the probe only hints |
| 2026-11-04 | Experiment power | t-test on system totals · cluster-week model | Cluster-week CV from hourly Poisson + estimated extra dispersion + assumed cluster noise | Switchback design cancels system-wide shocks within a week |
| 2026-11-04 | Backtest | random CV · walk-forward | Expanding window, 4-week horizon, 10 folds Mar–Dec 2012; 2 tuning folds; 2 warm-up folds only for calibration | Time-respecting; honest score on folds never used for tuning |
| 2026-11-04 | Evaluation hours | all rows · observed only | Observed hours only (inserted night zeros excluded from scores) | Scores must not reward predicting our own imputations |
| 2026-11-04 | CatBoost loss | Poisson · Tweedie p ∈ {1.2,1.4,1.6}; depth ∈ {6,8} | Chosen per segment by Poisson deviance on tuning folds | Count-aware loss; deviance is the proper score for the mean |
| 2026-11-04 | Weather at forecast time | observed · forecast | Observed (documented) | No archived weather forecasts in the data; production must use forecasts |
| 2026-11-04 | Conformal | quantile regression · split conformal (MAPIE) | MAPIE `SplitConformalRegressor` on a pass-through estimator over walk-forward OOS forecasts; gamma score on count+1; Mondrian by hour group if peaks under-covered | Distribution-free coverage, calibrated only on the past; relative score handles heteroscedastic counts; MAPIE 1.x has no built-in Mondrian, so groups are calibrated separately |
| 2026-11-04 | Predictive distribution for simulation | NB draws · conformal residual quantiles | Signed relative residual quantiles per hour group + Gaussian copula (ρ estimated from backtest) | Same calibration as the intervals; within-day correlation matters for the daily peak |
| 2026-11-04 | Concurrency | M/G/∞ formula · trip-level Monte Carlo | Exact trip-level simulation, vectorised with bincount/cumsum; common random numbers across trip medians | Literal to the blueprint; monotone in trip median by construction; validated against a slow reference |
| 2026-11-04 | Service level definition | share of hours · share of days | Share of days on which the fleet covers the daily peak (2011–12 day mix) | Matches "F(α) = α-quantile of daily peak concurrency" |
| 2026-11-04 | Missed-ride cost | full queue model · peak shortfall approximation | Peak shortfall × 60 / trip median | Placeholder costs anyway; stated as rough |
| 2026-11-04 | Desk numbers | live model calls · pre-computed cube | Demand cube in DuckDB, SQL shown with each answer | Fast, auditable, identical every time (blueprint 4.1) |
| 2026-11-04 | Desk without keys | require a key · offline rules mode | Deterministic rules router + SQL templates + templated writer | Desk usable and testable offline; LLM is an upgrade, not a dependency |
| 2026-11-04 | SQL safety | regex · sqlglot AST whitelist | sqlglot: one statement, SELECT-only, allowed tables/columns, no table functions, wrapped LIMIT 501 to detect truncation, read-only connection | Defence in depth (blueprint 4.2) |
| 2026-11-04 | Gradio version | 5.x · 6.x | Pinned `gradio>=5.20,<6` with `Chatbot(type="messages")` | Gradio 6 changed the Chatbot API; pin avoids silent breakage |
| 2026-11-04 | LLM default | Gemini · Claude · Ollama | Claude via LiteLLM (`anthropic/...` in config.yaml) | User preference; switchable in config |
| 2026-11-04 | Interval calibration trigger | Mondrian only if peaks under-covered · if peaks off by > 3 pts either way | Either way | Global conformal over-covered peaks (92% for an 80% interval) — intervals too wide where fleet decisions are made |
| 2026-11-04 | Build environment (this run) | wait for wheels · substitute | The cloud sandbox could not download DuckDB / CatBoost / Polars / vl-convert. The pipeline falls back automatically: pandas for Polars, scikit-learn HistGradientBoosting (Poisson) for CatBoost, vendored vega JS + headless Chromium for PNGs; storage ran on a SQLite stand-in for DuckDB outside the repo | `uv sync && make all` on a normal machine installs the real stack and the fallbacks switch off; the model card states which backend produced its numbers |
