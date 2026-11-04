"""Workstream 3.5 — fleet sizing: service level, newsvendor cost view, idle capacity, fleet vs flow."""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from .common import config, save_chart, save_csv, save_json


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    o = np.argsort(values)
    v, w = values[o], weights[o]
    cw = np.cumsum(w) / w.sum()
    return float(v[np.searchsorted(cw, q, side="left").clip(0, len(v) - 1)])


class FleetModel:
    """Holds the simulated daily-peak distribution for every (month, class, band, trip_median) cell and the
    empirical day mix (how often each cell occurs in 2011-12). Service level = share of days the fleet covers."""

    def __init__(self, peaks: pd.DataFrame, hourly: pd.DataFrame, weights: pd.DataFrame):
        # peaks: cell keys + trip_median + draw + peak ; hourly: keys + trip_median + hour + mean_in_use
        # weights: month, calendar_class, weather_band, n_days
        self.peaks, self.hourly, self.weights = peaks, hourly, weights
        keys = ["month", "calendar_class", "weather_band"]
        self._w = peaks.merge(weights, on=keys, how="inner")
        ndraw = self._w.groupby(keys + ["trip_median"]).peak.transform("size")
        self._w["w"] = self._w.n_days / ndraw
        hw = hourly.merge(weights, on=keys, how="inner")
        self._hw = hw

    def fleet(self, alpha: float, friction: float, trip_median: float) -> float:
        d = self._w[self._w.trip_median == trip_median]
        return float(np.ceil(weighted_quantile(d.peak.to_numpy(), d.w.to_numpy(), alpha) * friction))

    def peak_quantile(self, alpha: float, trip_median: float) -> float:
        d = self._w[self._w.trip_median == trip_median]
        return weighted_quantile(d.peak.to_numpy(), d.w.to_numpy(), alpha)

    def idle_share(self, fleet: float, trip_median: float) -> float:
        h = self._hw[self._hw.trip_median == trip_median]
        bike_hours = (h.mean_in_use * h.n_days).sum()
        return float(1 - bike_hours / (fleet * 24 * self.weights.n_days.sum()))

    def missed_rides_per_day(self, fleet: float, friction: float, trip_median: float) -> float:
        """Rough: bikes short at the daily peak x rides a bike can serve in one peak hour (60 / trip median)."""
        d = self._w[self._w.trip_median == trip_median]
        short = np.clip(d.peak.to_numpy() * friction - fleet, 0, None) / friction
        return float((short * d.w).sum() / d.w.sum() * 60 / trip_median)

    def utilisation(self, fleet: float, trip_median: float) -> pd.DataFrame:
        h = self._hw[self._hw.trip_median == trip_median]
        g = h.groupby(["calendar_class", "hour"]).apply(
            lambda x: pd.Series({"mean_in_use": np.average(x.mean_in_use, weights=x.n_days),
                                 "p95_in_use": np.average(x.p95_in_use, weights=x.n_days)}), include_groups=False).reset_index()
        g["fleet"] = fleet
        g["utilisation"] = g.mean_in_use / fleet
        g["idle_share"] = 1 - g.utilisation
        return g


def run(fm: FleetModel) -> dict:
    c, cc = config()["fleet"], config()["concurrency"]
    cost = c["cost"]
    trips = cc["trip_median_grid"]
    rows = []
    for a in sorted(set(c["service_levels"] + [c["recommended_service_level"]])):
        for f in c["friction_grid"]:
            for t in trips:
                F = fm.fleet(a, f, t)
                missed = fm.missed_rides_per_day(F, f, t)
                rows.append({"service_level": a, "friction": f, "trip_median": t, "peak_concurrency_q": fm.peak_quantile(a, t),
                             "required_fleet": F, "idle_share": fm.idle_share(F, t), "holding_cost_per_day": F * cost["bike_daily_co"],
                             "missed_rides_per_day": missed, "missed_ride_cost_per_day": missed * cost["missed_ride_cu"],
                             "cost_per_day": F * cost["bike_daily_co"] + missed * cost["missed_ride_cu"]})
    grid = pd.DataFrame(rows)
    save_csv(grid, "fleet_service_level_grid")

    f0, t0 = c["friction_default"], cc["trip_median_min"]
    nv = []
    for cu in cost["cu_grid"]:
        for co in cost["co_grid"]:
            a = cu / (cu + co)
            F = fm.fleet(a, f0, t0)
            nv.append({"cu": cu, "co": co, "optimal_service_level": a, "fleet": F, "idle_share": fm.idle_share(F, t0)})
    nv = pd.DataFrame(nv)
    save_csv(nv, "fleet_newsvendor")
    a_star = cost["missed_ride_cu"] / (cost["missed_ride_cu"] + cost["bike_daily_co"])
    F_cost = fm.fleet(a_star, f0, t0)

    a_rec = c["recommended_service_level"]
    F_rec = fm.fleet(a_rec, f0, t0)
    util = fm.utilisation(F_rec, t0)
    save_csv(util, "fleet_utilisation")
    flow = []
    for t in trips:
        hi, lo = fm.fleet(a_rec, 1.5, t), fm.fleet(a_rec, 1.2, t)
        flow.append({"trip_median": t, "fleet_poor_rebalancing": hi, "fleet_good_rebalancing": lo, "bikes_saved_by_rebalancing": hi - lo,
                     "pct_saved": (hi - lo) / hi})
    flow = pd.DataFrame(flow)
    save_csv(flow, "fleet_vs_flow")

    # figures
    d = fm._w[fm._w.trip_median == t0]
    hist = np.histogram(d.peak, bins=60, weights=d.w)
    hdf = pd.DataFrame({"peak": (hist[1][:-1] + hist[1][1:]) / 2, "share_of_days": hist[0] / hist[0].sum()})
    q95 = fm.peak_quantile(a_rec, t0)
    ch = alt.Chart(hdf).mark_bar(color="#1b6ca8").encode(x=alt.X("peak:Q", title="daily peak bikes in use (before friction)"), y=alt.Y("share_of_days:Q", axis=alt.Axis(format="%"), title="share of days"))
    rl = alt.Chart(pd.DataFrame({"x": [q95], "t": [f"95th pct = {q95:.0f}"]}))
    save_chart((ch + rl.mark_rule(color="#c0392b").encode(x="x:Q") + rl.mark_text(align="left", dx=4, dy=-120, color="#c0392b").encode(x="x:Q", text="t:N"))
               .properties(width=520, height=240, title=f"Daily peak concurrency distribution (trip median {t0} min, 2011-12 day mix)"), "ws3_concurrency_distribution")
    u = util[util.calendar_class.isin(["workday", "weekend"])]
    save_chart(alt.Chart(u).mark_line(point=True).encode(x=alt.X("hour:O"), y=alt.Y("utilisation:Q", axis=alt.Axis(format="%"), title=f"mean bikes in use / fleet of {F_rec:.0f}"), color="calendar_class:N")
               .properties(width=520, height=240, title="Hourly utilisation of the recommended fleet"), "ws3_utilisation_curve")
    sens = grid[(grid.service_level == a_rec) & (grid.friction == f0)][["trip_median", "required_fleet", "idle_share"]]
    out = {"recommended": {"service_level": a_rec, "friction": f0, "trip_median": t0, "fleet": F_rec,
                           "idle_share": fm.idle_share(F_rec, t0), "peak_q": q95},
           "cost_optimal": {"cu": cost["missed_ride_cu"], "co": cost["bike_daily_co"], "service_level": a_star, "fleet": F_cost},
           "fleet_vs_flow": flow.to_dict(orient="records"), "trip_sensitivity": sens.to_dict(orient="records"),
           "service_levels": grid[(grid.friction == f0) & (grid.trip_median == t0)][["service_level", "required_fleet"]].to_dict(orient="records"),
           "night_idle_3am_workday": float(util[(util.calendar_class == "workday") & (util.hour == 3)].idle_share.iloc[0]),
           "peak_utilisation_workday": float(util[util.calendar_class == "workday"].utilisation.max()),
           "share_hours_util_below_25pct_workday": float((util[util.calendar_class == "workday"].utilisation < 0.25).mean())}
    save_json(out, "ws3_fleet_summary")
    return out
