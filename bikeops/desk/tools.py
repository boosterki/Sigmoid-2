"""Desk tools: simulate_fleet, explain_driver, search_notes."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from ..common import config, connect


@lru_cache(maxsize=1)
def _fleet_model():
    from ..cube import load_fleet_model

    return load_fleet_model()


def simulate_fleet(service_level: float | None = None, friction: float | None = None, trip_median: float | None = None,
                   cu: float | None = None, co: float | None = None) -> dict:
    """Required fleet, idle share and cost per day. Reads the stored concurrency distribution (no live model call)."""
    c = config()
    a = service_level or c["fleet"]["recommended_service_level"]
    f = friction or c["desk"]["default_friction"]
    t = trip_median or c["desk"]["default_trip_median"]
    grid = c["concurrency"]["trip_median_grid"]
    t_used = min(grid, key=lambda g: abs(g - t))
    cu = c["fleet"]["cost"]["missed_ride_cu"] if cu is None else cu
    co = c["fleet"]["cost"]["bike_daily_co"] if co is None else co
    fm = _fleet_model()
    F = fm.fleet(a, f, t_used)
    missed = fm.missed_rides_per_day(F, f, t_used)
    util = fm.utilisation(F, t_used)
    res = {"service_level": a, "friction": f, "trip_median_requested": t, "trip_median_used": t_used,
           "required_fleet": F, "idle_share": fm.idle_share(F, t_used), "missed_rides_per_day": missed,
           "cost_per_day": F * co + missed * cu, "cu": cu, "co": co,
           "fleet_poor_rebalancing": fm.fleet(a, 1.5, t_used), "fleet_good_rebalancing": fm.fleet(a, 1.2, t_used)}
    res["bikes_saved_by_rebalancing"] = res["fleet_poor_rebalancing"] - res["fleet_good_rebalancing"]
    res["utilisation"] = util
    return res


def explain_driver(feature: str | None = None, con=None) -> pd.DataFrame:
    own = con is None
    con = con or connect(read_only=True)
    try:
        q = "SELECT segment, feature, mean_abs_shap, rank FROM shap_summary"
        if feature:
            return con.execute(q + " WHERE feature = ? ORDER BY segment", [feature]).df()
        return con.execute(q + " WHERE rank <= 6 ORDER BY segment, rank").df()
    finally:
        if own:
            con.close()


def search_notes(query: str, k: int = 3, con=None) -> pd.DataFrame:
    """DuckDB full-text search (BM25) over docs/*.md and Readme.txt; keyword fallback if FTS is unavailable."""
    own = con is None
    con = con or connect(read_only=True)
    try:
        try:
            con.execute("LOAD fts")
            df = con.execute(
                "SELECT note_id, source, heading, text, fts_main_notes.match_bm25(note_id, ?) AS score FROM notes "
                "WHERE score IS NOT NULL ORDER BY score DESC LIMIT ?", [query, k]).df()
            if len(df):
                return df
        except Exception:
            pass
        notes = con.execute("SELECT note_id, source, heading, text FROM notes").df()
        terms = [t for t in query.lower().split() if len(t) > 2]
        notes["score"] = notes.apply(lambda r: sum((r.text + " " + r.heading).lower().count(t) for t in terms), axis=1)
        return notes[notes.score > 0].sort_values("score", ascending=False).head(k)
    finally:
        if own:
            con.close()
