"""Text-to-SQL with guardrails.

LLM writes SQL over whitelisted views only; sqlglot parses it -> single SELECT -> allowed tables/columns ->
row cap. Execution happens on a read-only DuckDB connection. One self-correction round on error.
Offline ('rules') mode builds the same SQL from resolved slots with templates.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import sqlglot
from sqlglot import exp

from ..common import config, connect
from . import llm
from .resolver import Slots

SEMANTIC = {
    "demand_cube": {
        "desc": "Forecast plan at the 2012 demand level. One row per month x calendar_class x hour x weather_band x trip_median. "
                "Use trip_median = 18 unless the user gives trip length.",
        "columns": {
            "month": "1-12", "calendar_class": "'workday' | 'weekend' | 'holiday' | 'bridge'", "hour": "0-23 (start of hour)",
            "weather_band": "'dry-mild' | 'dry-hot' (>=28 C) | 'dry-cold' (<10 C) | 'wet' (rain/snow)",
            "trip_median": "12 | 18 | 25 minutes (assumed median trip duration)",
            "casual_p10": "rentals by non-members, 10th pct", "casual_p50": "median forecast", "casual_p90": "90th pct",
            "registered_p10": "members, 10th pct", "registered_p50": "members, median", "registered_p90": "members, 90th pct",
            "total_p10": "all rentals 10th pct", "total_p50": "all rentals median forecast", "total_p90": "all rentals 90th pct",
            "bikes_in_use_p50": "median of the hour's peak bikes on the road", "bikes_in_use_p95": "95th pct of bikes on the road",
            "bikes_95_service": "bikes that should be available in that hour for a 95% service level incl. spatial friction 1.3",
            "temp_c": "representative temperature used", "weather_support_hours": "historical hours behind the weather profile",
        }},
    "history_hourly": {
        "desc": "Cleaned observed hours 2011-2012 (imputed_row = 1 marks night-lull zero hours added in cleaning).",
        "columns": {"date": "DATE", "hour": "0-23", "month": "1-12", "year": "2011 | 2012",
                    "calendar_class": "'workday' | 'weekend' | 'holiday' | 'bridge'", "weather": "'clear' | 'mist' | 'wet'",
                    "weather_band": "as in demand_cube", "wet": "1 if rain/snow", "temp_c": "deg C", "hum_pct": "%",
                    "wind_kmh": "km/h", "casual": "rentals", "registered": "rentals", "cnt": "casual + registered",
                    "archetype": "day shape cluster", "imputed_row": "0/1", "disruption_day": "0/1"}},
    "history_daily": {
        "desc": "One row per day 2011-2012.",
        "columns": {"date": "DATE", "year": "2011|2012", "month": "1-12", "weekday": "0=Sunday..6", "calendar_class": "as above",
                    "weather": "'clear'|'mist'|'wet' (worst of day)", "weather_band": "day weather band", "temp_c": "mean deg C",
                    "casual": "rentals", "registered": "rentals", "cnt": "total rentals", "archetype": "day shape cluster",
                    "disruption_day": "1 if anomalous day (STL robust z > 3.5)", "robust_z": "robust z of STL remainder",
                    "stl_trend": "trend level (rentals/day)"}},
    "fleet_scenarios": {
        "desc": "Required fleet by service level x friction x trip median (2011-12 day mix).",
        "columns": {"service_level": "0.90|0.95|0.99", "friction": "1.2|1.3|1.4|1.5 (bikes stranded at the wrong docks)",
                    "trip_median": "12|18|25", "peak_concurrency_q": "quantile of daily peak bikes in use", "required_fleet": "bikes",
                    "idle_share": "share of bike-hours idle", "holding_cost_per_day": "USD placeholder", "missed_rides_per_day": "rough",
                    "missed_ride_cost_per_day": "USD placeholder", "cost_per_day": "USD placeholder"}},
    "fleet_utilisation": {
        "desc": "Hourly utilisation of the recommended fleet (95% service, friction 1.3, trip median 18).",
        "columns": {"calendar_class": "as above", "hour": "0-23", "mean_in_use": "mean bikes on the road",
                    "p95_in_use": "95th pct bikes on the road", "fleet": "recommended fleet", "utilisation": "mean_in_use / fleet",
                    "idle_share": "1 - utilisation"}},
    "fleet_vs_flow": {
        "desc": "Bikes required at 95% service with poor (1.5) vs good (1.2) rebalancing.",
        "columns": {"trip_median": "minutes", "fleet_poor_rebalancing": "bikes", "fleet_good_rebalancing": "bikes",
                    "bikes_saved_by_rebalancing": "difference", "pct_saved": "share"}},
    "hypothesis_results": {
        "desc": "Weather sensitivity test (Workstream 2): ratio of rate-ratios casual/registered for wet weather.",
        "columns": {"analysis": "text", "method": "'NB GLM' | 'Matched wet vs dry'", "rrr": "ratio of rate-ratios", "ci_low": "95% CI",
                    "ci_high": "95% CI", "p_value": "p", "rr_casual": "wet/dry rate ratio casual", "rr_registered": "wet/dry rate ratio members"}},
    "temperature_elasticity": {
        "desc": "% change in rentals per +1 C in the 5-25 C band.",
        "columns": {"segment": "'casual'|'registered'", "pct_per_degC": "%", "ci_low": "%", "ci_high": "%", "band_c": "text", "n_obs": "n"}},
    "shap_summary": {"desc": "Drivers of demand: mean |SHAP| per feature and segment.",
                     "columns": {"segment": "casual|registered", "feature": "name", "mean_abs_shap": "importance", "rank": "1 = top"}},
}
ALLOWED = {t: set(v["columns"]) for t, v in SEMANTIC.items()}

EXAMPLES = [
    ("How many bikes should be available at 8 AM on a dry mild workday in May?",
     "SELECT hour, bikes_95_service, total_p50, total_p90 FROM demand_cube WHERE month = 5 AND calendar_class = 'workday' "
     "AND weather_band = 'dry-mild' AND hour = 8 AND trip_median = 18"),
    ("Which hours on a cold Sunday in January need the most bikes?",
     "SELECT hour, bikes_95_service, total_p50 FROM demand_cube WHERE month = 1 AND calendar_class = 'weekend' "
     "AND weather_band = 'dry-cold' AND trip_median = 18 ORDER BY bikes_95_service DESC LIMIT 5"),
    ("Members vs casual riders on weekends in 2012?",
     "SELECT calendar_class, SUM(casual) AS casual, SUM(registered) AS registered FROM history_hourly WHERE year = 2012 "
     "AND calendar_class = 'weekend' GROUP BY calendar_class"),
    ("What was the busiest day?", "SELECT date, cnt, weather, calendar_class FROM history_daily ORDER BY cnt DESC LIMIT 1"),
    ("How much busier was 2012 than 2011?", "SELECT year, SUM(cnt) AS rentals FROM history_daily GROUP BY year ORDER BY year"),
    ("What share of the fleet is idle at 5 AM on weekends?",
     "SELECT hour, idle_share, mean_in_use, fleet FROM fleet_utilisation WHERE calendar_class = 'weekend' AND hour = 5"),
]


class SQLRejected(ValueError):
    pass


BANNED = tuple(getattr(exp, n) for n in ["Insert", "Update", "Delete", "Drop", "Create", "Alter", "Command", "Merge",
                                         "Copy", "Attach", "Detach", "Pragma", "Set", "Use", "Transaction", "Commit",
                                         "Rollback", "TruncateTable", "Grant", "LoadData", "Into"] if hasattr(exp, n))


def guard(sql: str, max_rows: int | None = None) -> str:
    """Parse -> exactly one SELECT -> only whitelisted tables/columns -> no table functions -> row cap."""
    max_rows = max_rows or config()["desk"]["max_rows"]
    try:
        stmts = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
    except sqlglot.errors.ParseError as e:
        raise SQLRejected(f"could not parse SQL: {e}") from e
    if len(stmts) != 1:
        raise SQLRejected("exactly one statement is allowed")
    tree = stmts[0]
    if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except, exp.Subquery)):
        raise SQLRejected(f"only SELECT statements are allowed (got {type(tree).__name__})")
    for node in tree.walk():
        if isinstance(node, BANNED):
            raise SQLRejected(f"statement type not allowed: {type(node).__name__}")
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    used = set()
    for t in tree.find_all(exp.Table):
        if not isinstance(t.this, exp.Identifier):
            raise SQLRejected("table functions are not allowed")
        if t.args.get("db") or t.args.get("catalog"):
            raise SQLRejected("qualified (schema/catalog) table names are not allowed")
        name = t.name.lower()
        if name in ctes:
            continue
        if name not in ALLOWED:
            raise SQLRejected(f"unknown table: {name}")
        used.add(name)
    if not used and not ctes:
        raise SQLRejected("query must read from an allowed table")
    allowed_cols = set().union(*[ALLOWED[t] for t in used]) if used else set()
    aliases = {a.alias.lower() for a in tree.find_all(exp.Alias)}
    aliases |= {c.alias_or_name.lower() for c in tree.find_all(exp.TableAlias)}
    for cte in tree.find_all(exp.CTE):
        for col in cte.this.find_all(exp.Alias):
            aliases.add(col.alias.lower())
    for c in tree.find_all(exp.Column):
        n = c.name.lower()
        if n and n not in allowed_cols and n not in aliases:
            raise SQLRejected(f"unknown column: {c.name}")
    for f in tree.find_all(exp.Anonymous):
        if f.name.lower().startswith(("read_", "glob", "sniff", "duckdb_", "pragma_")):
            raise SQLRejected(f"function not allowed: {f.name}")
    inner = tree.sql(dialect="duckdb")
    return f"SELECT * FROM ({inner}) AS q LIMIT {max_rows + 1}"


@dataclass
class SQLResult:
    sql: str
    rows: pd.DataFrame
    truncated: bool
    error: str | None = None
    attempts: int = 1


def execute(sql: str, max_rows: int | None = None, con=None) -> SQLResult:
    max_rows = max_rows or config()["desk"]["max_rows"]
    safe = guard(sql, max_rows)
    own = con is None
    con = con or connect(read_only=True)
    try:
        df = con.execute(safe).df()
    finally:
        if own:
            con.close()
    trunc = len(df) > max_rows
    return SQLResult(sql=sql, rows=df.head(max_rows), truncated=trunc)


def semantic_prompt() -> str:
    lines = ["You write DuckDB SQL for a bike-share operations desk. Use ONLY these tables and columns:"]
    for t, v in SEMANTIC.items():
        lines.append(f"\nTABLE {t}: {v['desc']}")
        for c, d in v["columns"].items():
            lines.append(f"  - {c}: {d}")
    lines.append("\nRules: one SELECT statement; no DDL/DML; at most 500 rows; quote string values exactly as listed; "
                 "use trip_median = 18 when not specified; dates as DATE 'YYYY-MM-DD'.")
    lines.append("\nExamples:")
    for q, s in EXAMPLES:
        lines.append(f"Q: {q}\nSQL: {s}")
    lines.append('\nReturn {"sql": "<query>"}.')
    return "\n".join(lines)


def llm_sql(question: str, slots: Slots, provider: str | None = None, con=None) -> SQLResult:
    hint = f"Question: {question}\nResolved slots (trust these): {slots.dict()}"
    out = llm.complete(semantic_prompt(), hint, provider)
    sql = out.get("sql", "")
    try:
        return execute(sql, con=con)
    except Exception as e:  # self-check: one corrected attempt
        out2 = llm.complete(semantic_prompt(), f"{hint}\nYour previous SQL:\n{sql}\nfailed with: {e}\nReturn a corrected query.", provider)
        sql2 = out2.get("sql", "")
        try:
            r = execute(sql2, con=con)
            r.attempts = 2
            return r
        except Exception as e2:
            return SQLResult(sql=sql2, rows=pd.DataFrame(), truncated=False, error=str(e2), attempts=2)


# ------------------------------------------------------------------------------------- rules (offline) SQL
def rules_sql(question: str, intent: str, slots: Slots) -> str | None:
    ql = question.lower()
    tm = int(slots.trip_median) if slots.trip_median in (12, 18, 25) else 18
    if intent == "lookup":
        if "idle" in ql or "utilis" in ql or "utiliz" in ql:
            cls = slots.calendar_class or "workday"
            where = f"calendar_class = '{cls}'" + (f" AND hour = {slots.hour}" if slots.hour is not None else "")
            return f"SELECT hour, idle_share, utilisation, mean_in_use, fleet FROM fleet_utilisation WHERE {where} ORDER BY hour"
        m = slots.month or 6
        cls = slots.calendar_class or "workday"
        band = slots.weather_band or "dry-mild"
        base = (f"FROM demand_cube WHERE month = {m} AND calendar_class = '{cls}' AND weather_band = '{band}' "
                f"AND trip_median = {tm}")
        if slots.hour is not None:
            return f"SELECT hour, bikes_95_service, bikes_in_use_p95, total_p10, total_p50, total_p90 {base} AND hour = {slots.hour}"
        return f"SELECT hour, bikes_95_service, bikes_in_use_p95, total_p50, total_p90 {base} ORDER BY bikes_95_service DESC LIMIT 5"
    if intent == "compare":
        if "wet" in ql or "rain" in ql or "dry" in ql:
            return ("SELECT CASE WHEN wet = 1 THEN 'wet' ELSE 'dry' END AS weather_group, AVG(casual) AS avg_casual, "
                    "AVG(registered) AS avg_registered, COUNT(*) AS hours FROM history_hourly WHERE imputed_row = 0 "
                    "GROUP BY weather_group ORDER BY weather_group")
        if "2011" in ql or "2012" in ql or "year" in ql:
            return "SELECT year, SUM(casual) AS casual, SUM(registered) AS registered, SUM(cnt) AS cnt FROM history_daily GROUP BY year ORDER BY year"
        if "weekend" in ql or "weekday" in ql:
            return ("SELECT calendar_class, AVG(cnt) AS avg_daily_rentals, AVG(casual) AS avg_casual, AVG(registered) AS avg_registered "
                    "FROM history_daily GROUP BY calendar_class ORDER BY calendar_class")
        return "SELECT year, SUM(casual) AS casual, SUM(registered) AS registered FROM history_daily GROUP BY year ORDER BY year"
    if intent == "history":
        if slots.date:
            return (f"SELECT date, cnt, casual, registered, weather, calendar_class, disruption_day, robust_z, stl_trend "
                    f"FROM history_daily WHERE date = DATE '{slots.date}'")
        if "lowest" in ql or "quietest" in ql:
            return "SELECT date, cnt, weather, calendar_class, disruption_day FROM history_daily ORDER BY cnt ASC LIMIT 5"
        if "highest" in ql or "busiest" in ql or "record" in ql:
            return "SELECT date, cnt, weather, calendar_class FROM history_daily ORDER BY cnt DESC LIMIT 5"
        where = []
        if slots.year:
            where.append(f"year = {slots.year}")
        if slots.month:
            where.append(f"month = {slots.month}")
        w = (" WHERE " + " AND ".join(where)) if where else ""
        return f"SELECT year, month, SUM(cnt) AS rentals, AVG(cnt) AS avg_per_day, SUM(disruption_day) AS disruption_days FROM history_daily{w} GROUP BY year, month ORDER BY year, month"
    return None
