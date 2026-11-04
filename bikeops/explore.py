"""Workstream 1.3 — understanding the demand signal (STL, archetypes, segments, weather, concentration)."""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from . import archetypes as arch
from .common import read_table, save_chart, save_csv, save_json

alt.data_transformers.disable_max_rows()
SEG_COLORS = alt.Scale(domain=["casual", "registered"], range=["#d95f02", "#1b6ca8"])
W, H = 560, 220


def fig_stl(stl: pd.DataFrame):
    d = stl.assign(date=pd.to_datetime(stl.dteday))
    long = d.melt(id_vars=["date", "disruption_day"], value_vars=["log_cnt", "stl_trend", "stl_weekly", "stl_remainder"],
                  var_name="component", value_name="value")
    names = {"log_cnt": "1 observed log(daily rentals)", "stl_trend": "2 trend", "stl_weekly": "3 weekly cycle",
             "stl_remainder": "4 remainder (red = disruption day)"}
    long["component"] = long.component.map(names)
    base = alt.Chart(long).encode(x=alt.X("date:T", title=None))
    line = base.mark_line(strokeWidth=0.8, color="#333").encode(y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)))
    pts = base.transform_filter("datum.disruption_day == 1").mark_point(color="#c0392b", filled=True, size=30).encode(y="value:Q")
    return (line + pts).properties(width=W, height=90).facet(row=alt.Row("component:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left"))).resolve_scale(y="independent").properties(title="STL decomposition of daily rentals (log scale, weekly period)")


def fig_centroids(cent: pd.DataFrame, counts: pd.Series):
    long = cent.reset_index(names="archetype").melt(id_vars="archetype", var_name="hour", value_name="share")
    long["hour"] = long.hour.astype(int)
    long["label"] = long.archetype.map(lambda a: f"{a} (n={int(counts.get(a, 0))} days)")
    return alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("hour:O", title="hour of day"), y=alt.Y("share:Q", title="share of the day's rentals", axis=alt.Axis(format="%")),
        color=alt.Color("label:N", title="archetype"),
    ).properties(width=W, height=H, title="Day archetypes — KMeans centroids of the 24-hour shape")


def fig_segments(sig: pd.DataFrame):
    d = sig.assign(day_type=np.where(sig.workingday == 1, "working day", "non-working day"))
    m = d.groupby(["day_type", "hour"])[["casual", "registered"]].mean().reset_index().melt(
        id_vars=["day_type", "hour"], var_name="segment", value_name="mean_rentals")
    return alt.Chart(m).mark_line(point=True).encode(
        x=alt.X("hour:O"), y=alt.Y("mean_rentals:Q", title="mean rentals per hour"),
        color=alt.Color("segment:N", scale=SEG_COLORS), column=alt.Column("day_type:N", title=None),
    ).properties(width=300, height=H, title="Segment hour profiles")


def fig_temperature(tr: pd.DataFrame):
    long = tr.melt(id_vars=["temp_bin_c", "hours"], value_vars=["casual", "registered"], var_name="segment", value_name="mean_rentals")
    return alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("temp_bin_c:Q", title="temperature (°C, 2° bins)"), y=alt.Y("mean_rentals:Q", title="mean rentals per hour"),
        color=alt.Color("segment:N", scale=SEG_COLORS),
    ).properties(width=W, height=H, title="Temperature response by segment (rise, then hot-weather plateau)")


def fig_concentration(lor: pd.DataFrame, top10: float):
    base = alt.Chart(lor).encode(x=alt.X("hour_share:Q", title="share of the week's hours (busiest first)", axis=alt.Axis(format="%")))
    curve = base.mark_line(color="#1b6ca8", strokeWidth=2.5).encode(y=alt.Y("rental_share:Q", title="share of the week's rentals", axis=alt.Axis(format="%")))
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color="#999").encode(x="x:Q", y="y:Q")
    rule = alt.Chart(pd.DataFrame({"x": [0.1]})).mark_rule(color="#c0392b").encode(x="x:Q")
    txt = alt.Chart(pd.DataFrame({"x": [0.11], "y": [top10 - 0.05], "t": [f"busiest 10% of hours = {top10:.0%} of rentals"]})).mark_text(align="left", color="#c0392b").encode(x="x:Q", y="y:Q", text="t:N")
    return (diag + curve + rule + txt).properties(width=W, height=300, title="Concentration of weekly rentals (Lorenz-style)")


def concentration(sig: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    d = sig.assign(week=sig.ts.dt.to_period("W-SUN"))
    sizes = d.groupby("week").size()
    full = sizes[sizes == 168].index
    tops, p2a = [], []
    grid = np.linspace(0, 1, 169)
    curves = []
    for _w, g in d[d.week.isin(full)].groupby("week"):
        v = np.sort(g.cnt.to_numpy())[::-1]
        if v.sum() == 0:
            continue
        cs = np.r_[0, np.cumsum(v)] / v.sum()
        curves.append(cs)
        tops.append(cs[17])  # top 10% of 168 hours ~ 17 hours
        p2a.append(v.max() / v.mean())
    curves = np.array(curves)
    lor = pd.DataFrame({"hour_share": grid, "rental_share": curves.mean(axis=0)})
    stats = {"weeks": len(curves), "top10pct_hours_share_mean": float(np.mean(tops)),
             "top10pct_hours_share_p10": float(np.quantile(tops, 0.1)), "top10pct_hours_share_p90": float(np.quantile(tops, 0.9)),
             "top20pct_hours_share_mean": float(curves[:, 34].mean()), "half_of_rentals_in_pct_hours": float(grid[np.argmax(curves.mean(0) >= 0.5)]),
             "weekly_peak_to_average_mean": float(np.mean(p2a))}
    return lor, stats


def run(sig: pd.DataFrame, model: arch.ArchetypeModel, labels: pd.Series, k_scores: pd.DataFrame) -> dict:
    stl = read_table("stl_daily")
    day = read_table("day_clean")
    save_chart(fig_stl(stl), "ws1_stl_panel")

    # ---- archetypes
    di = arch.day_info(sig).merge(labels, left_on="dteday", right_index=True, how="left")
    di = di.merge(day[["dteday", "weathersit", "disruption_day"]], on="dteday", how="left")
    counts = di.archetype.value_counts()
    save_chart(fig_centroids(model.centroids, counts), "ws1_archetype_centroids")
    save_csv(k_scores, "archetype_silhouette")
    save_csv(model.centroids.reset_index(names="archetype"), "archetype_centroids")
    xt_wd = pd.crosstab(di.archetype, di.workingday.map({1: "working day", 0: "non-working day"}))
    xt_ws = pd.crosstab(di.archetype, di.weathersit.map({1: "clear", 2: "mist", 3: "wet"}))
    xt_m = pd.crosstab(di.archetype, di.month)
    for n, t in [("archetype_x_workingday", xt_wd), ("archetype_x_weathersit", xt_ws), ("archetype_x_month", xt_m)]:
        save_csv(t.reset_index(), n)
    expected = np.where(di.workingday == 1, "commute", "leisure")
    base_name = di.archetype.fillna("").str.split("-").str[-1].where(~di.archetype.fillna("").str.startswith("flattened"), "flattened")
    mis = di[(di.archetype.notna()) & (((di.workingday == 1) & ~di.archetype.str.contains("commute", na=False)) |
                                      ((di.workingday == 0) & di.archetype.str.contains("commute", na=False)))].copy()
    import holidays as hol
    us = hol.US(years=[2011, 2012])

    def explain(r):
        reasons = []
        if r.calendar_class == "bridge":
            reasons.append("bridge day next to a holiday")
        if r.calendar_class == "holiday":
            reasons.append(f"holiday ({us.get(pd.Timestamp(r.date).date(), 'holiday')})")
        if r.weathersit == 3:
            reasons.append("wet weather flattens the commute peaks")
        if r.disruption_day == 1:
            reasons.append("disruption day (STL outlier)")
        dt = pd.Timestamp(r.date)
        if (dt.month == 12 and dt.day >= 20) or (dt.month == 1 and dt.day <= 3):
            reasons.append("Christmas / New Year week")
        if r.workingday == 0 and "commute" in str(r.archetype) and not reasons:
            reasons.append("non-working day with residual commute traffic (e.g. federal-only holiday, weekday-like Saturday)")
        nxt = dt + pd.Timedelta(days=1)
        prv = dt - pd.Timedelta(days=1)
        if not reasons and (nxt.date() in us or prv.date() in us):
            reasons.append("adjacent to a federal holiday")
        return "; ".join(reasons) or "no obvious calendar or weather reason"

    mis["explanation"] = mis.apply(explain, axis=1)
    mis = mis[["dteday", "calendar_class", "workingday", "weathersit", "archetype", "explanation"]]
    save_csv(mis, "archetype_misfits")
    _ = expected, base_name

    # ---- segments
    seg_arch = sig.merge(labels, left_on="dteday", right_index=True).groupby("archetype")[["casual", "registered"]].sum()
    seg_arch["casual_share"] = seg_arch.casual / (seg_arch.casual + seg_arch.registered)
    save_csv(seg_arch.reset_index(), "segment_share_by_archetype")
    seg_hour = sig.assign(day_type=np.where(sig.workingday == 1, "working", "non-working")).groupby(["day_type", "hour"])[["casual", "registered"]].sum()
    seg_hour["casual_share"] = seg_hour.casual / (seg_hour.casual + seg_hour.registered)
    save_csv(seg_hour.reset_index(), "segment_share_by_hour")
    save_chart(fig_segments(sig), "ws1_segment_hour_profiles")

    # ---- weather response
    obs = sig[sig.imputed_row == 0]
    tr = obs.assign(temp_bin_c=(np.floor(obs.temp_c / 2) * 2 + 1)).groupby("temp_bin_c").agg(
        casual=("casual", "mean"), registered=("registered", "mean"), hours=("cnt", "size")).reset_index()
    tr = tr[tr.hours >= 30]
    save_csv(tr, "temperature_response")
    save_chart(fig_temperature(tr), "ws1_temperature_response")
    ws = obs.groupby("weathersit")[["casual", "registered", "cnt"]].mean()
    ws_rel = ws / ws.loc[1]
    ws_tab = ws.join(ws_rel, rsuffix="_rel_to_clear").reset_index()
    save_csv(ws_tab, "demand_by_weathersit")

    # ---- concentration
    lor, cstats = concentration(sig)
    save_csv(lor, "concentration_curve")
    save_chart(fig_concentration(lor, cstats["top10pct_hours_share_mean"]), "ws1_concentration_curve")

    peak_t = tr.loc[tr.registered.idxmax(), "temp_bin_c"]
    stats = {
        "k": model.k, "silhouette": k_scores.set_index("k").silhouette.round(3).to_dict(),
        "archetype_counts": counts.to_dict(),
        "archetype_x_workingday": xt_wd.to_dict(),
        "n_misfits": int(len(mis)),
        "casual_share_by_archetype": seg_arch.casual_share.round(3).to_dict(),
        "casual_share_overall": float(sig.casual.sum() / sig.cnt.sum()),
        "registered_peak_temp_bin_c": float(peak_t),
        "casual_peak_temp_bin_c": float(tr.loc[tr.casual.idxmax(), "temp_bin_c"]),
        "wet_vs_clear_casual": float(ws_rel.loc[3, "casual"]), "wet_vs_clear_registered": float(ws_rel.loc[3, "registered"]),
        "concentration": cstats,
    }
    save_json(stats, "ws1_signal_stats")
    return stats
