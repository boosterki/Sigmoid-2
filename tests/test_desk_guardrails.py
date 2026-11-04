"""WS4 guardrails: only single SELECTs over whitelisted tables/columns; long results truncated."""
import pytest

from bikeops.desk.sql_agent import SQLRejected, execute, guard

from .conftest import require

BAD = [
    "DROP TABLE demand_cube",
    "UPDATE demand_cube SET total_p50 = 0",
    "DELETE FROM history_daily",
    "INSERT INTO notes VALUES (1, 'x', 'y', 'z')",
    "ATTACH 'other.duckdb' AS o",
    "CREATE TABLE x AS SELECT * FROM demand_cube",
    "COPY demand_cube TO 'out.csv'",
    "PRAGMA database_list",
    "SELECT * FROM demand_cube; DROP TABLE demand_cube",
    "SELECT * FROM raw_hour",
    "SELECT * FROM information_schema.tables",
    "SELECT * FROM read_csv_auto('raw/hour.csv')",
    "SELECT secret_column FROM demand_cube",
    "SELECT * FROM main.demand_cube",
]


@pytest.mark.parametrize("sql", BAD)
def test_rejected(sql):
    with pytest.raises(SQLRejected):
        guard(sql)


def test_allowed_select_gets_row_cap():
    s = guard("SELECT hour, total_p50 FROM demand_cube WHERE month = 7")
    assert s.endswith("LIMIT 501")


def test_aliases_and_ctes_allowed():
    guard("WITH w AS (SELECT wet, AVG(casual) AS c FROM history_hourly GROUP BY wet) SELECT c FROM w")


def test_long_results_truncated():
    require("history_hourly")
    r = execute("SELECT date, hour, cnt FROM history_hourly")
    assert r.truncated and len(r.rows) == 500
