"""WS4 acceptance set (blueprint 4.4): each answer must match direct SQL on the cube / warehouse.

Runs with the offline rules provider always, and with each LLM provider that is configured
(ANTHROPIC_API_KEY / GEMINI_API_KEY in .env, or a running Ollama)."""
import pandas as pd
import pytest

from bikeops.common import connect
from bikeops.desk import llm
from bikeops.desk.core import Desk

from .conftest import require

PROVIDERS = ["rules", "anthropic", "gemini", "ollama"]


def q1(sql, params=None):
    con = connect(read_only=True)
    try:
        return con.execute(sql, params or []).df()
    finally:
        con.close()


@pytest.fixture(scope="module", params=PROVIDERS)
def desk(request):
    require("demand_cube", "fleet_scenarios", "notes", "concurrency_peaks")
    if not llm.available(request.param):
        pytest.skip(f"{request.param} not configured")
    return Desk(request.param)


def _num_in(ans, value, tol=0.5):
    """value appears in result rows (any numeric cell) within tol."""
    import numpy as np

    vals = ans.rows.select_dtypes("number").to_numpy().ravel()
    return bool(np.any(np.abs(vals - value) <= tol))


def test_a_rainy_weekday_november_6pm(desk):
    a = desk.ask("How many bikes should be available at 6 PM on a rainy weekday in November?")
    exp = q1("SELECT bikes_95_service FROM demand_cube WHERE month=11 AND calendar_class='workday' AND weather_band='wet' "
             "AND hour=18 AND trip_median=18").iloc[0, 0]
    assert a.intent == "lookup" and _num_in(a, exp)


def test_b_hot_saturday_july_hours(desk):
    a = desk.ask("Which hours on a hot Saturday in July need the most bikes?")
    exp = q1("SELECT hour, bikes_95_service FROM demand_cube WHERE month=7 AND calendar_class='weekend' AND weather_band='dry-hot' "
             "AND trip_median=18 ORDER BY bikes_95_service DESC LIMIT 3")
    assert a.intent == "lookup"
    assert set(exp.hour) <= set(a.rows.hour.astype(int)) and _num_in(a, exp.bikes_95_service.iloc[0])


def test_c_casual_wet_vs_dry(desk):
    a = desk.ask("How much smaller is casual demand on wet days compared with dry ones?")
    exp = q1("SELECT AVG(casual) FILTER (WHERE wet=1) / AVG(casual) FILTER (WHERE wet=0) FROM history_hourly WHERE imputed_row=0").iloc[0, 0]
    assert a.intent == "compare"
    r = a.rows
    col = [c for c in r.columns if "casual" in c.lower()][0]
    grp = [c for c in r.columns if not pd.api.types.is_numeric_dtype(r[c])][0]
    ratio = r.loc[r[grp].astype(str).str.lower().str.contains("wet") | (r[grp].astype(str) == "1"), col].iloc[0] / \
        r.loc[~(r[grp].astype(str).str.lower().str.contains("wet") | (r[grp].astype(str) == "1")), col].iloc[0]
    assert abs(ratio - exp) < 0.01


def test_d_fleet_95_trips_25(desk):
    from bikeops.cube import load_fleet_model

    a = desk.ask("What fleet do we need for a 95% service level if trips average 25 minutes?")
    exp = load_fleet_model().fleet(0.95, 1.3, 25)
    grid = q1("SELECT required_fleet FROM fleet_scenarios WHERE service_level=0.95 AND friction=1.3 AND trip_median=25").iloc[0, 0]
    assert a.intent == "what_if_fleet" and exp == grid and _num_in(a, exp)


def test_e_rebalancing_saving(desk):
    a = desk.ask("How many bikes would better rebalancing save us?")
    exp = q1("SELECT bikes_saved_by_rebalancing FROM fleet_vs_flow WHERE trip_median=18").iloc[0, 0]
    assert a.intent == "what_if_fleet" and _num_in(a, exp)


def test_f_29_october_2012(desk):
    a = desk.ask("What happened on 29 October 2012?")
    exp = q1("SELECT cnt FROM history_daily WHERE date = DATE '2012-10-29'").iloc[0, 0]
    assert a.intent == "history" and _num_in(a, exp) and exp == 22
    assert "sandy" in (a.text + " ".join(a.values.get("notes", []))).lower()


def test_g_windspeed_zero(desk):
    a = desk.ask("Why do you treat windspeed zero as missing?")
    assert a.intent == "explain"
    txt = " ".join(a.rows.text.astype(str)).lower() if "text" in a.rows else a.text.lower()
    assert "windspeed" in txt and ("sentinel" in txt or "missing" in txt or "calm" in txt)


def test_h_idle_3am_weekday(desk):
    a = desk.ask("What share of the fleet sits idle at 3 AM on a weekday?")
    exp = q1("SELECT idle_share FROM fleet_utilisation WHERE calendar_class='workday' AND hour=3").iloc[0, 0]
    assert a.intent == "lookup" and _num_in(a, exp, tol=1e-6)
