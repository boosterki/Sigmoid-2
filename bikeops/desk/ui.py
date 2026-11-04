"""Gradio app — three tabs: Ask the desk, Demand explorer, Fleet simulator.  Run: make app"""
from __future__ import annotations

import altair as alt
import pandas as pd

from ..common import config, connect
from .core import Answer, Desk

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def answer_chart(a: Answer):
    r = a.rows
    if r is None or r.empty:
        return None
    if "hour" in r.columns and "bikes_95_service" in r.columns and len(r) > 1:
        return alt.Chart(r).mark_bar(color="#1b6ca8").encode(x="hour:O", y=alt.Y("bikes_95_service:Q", title="bikes at 95% service")).properties(width=420, height=220)
    if "weather_group" in r.columns:
        long = r.melt(id_vars="weather_group", value_vars=["avg_casual", "avg_registered"], var_name="segment", value_name="mean_rentals_per_hour")
        return alt.Chart(long).mark_bar().encode(x="segment:N", y="mean_rentals_per_hour:Q", color="weather_group:N", xOffset="weather_group:N").properties(width=380, height=220)
    if "utilisation" in a.values and isinstance(a.values["utilisation"], pd.DataFrame):
        u = a.values["utilisation"]
        u = u[u.calendar_class.isin(["workday", "weekend"])]
        return alt.Chart(u).mark_line(point=True).encode(x="hour:O", y=alt.Y("mean_in_use:Q", title="bikes in use (mean)"), color="calendar_class:N").properties(width=420, height=220)
    if "idle_share" in r.columns and "hour" in r.columns:
        return alt.Chart(r).mark_bar(color="#7f8c8d").encode(x="hour:O", y=alt.Y("idle_share:Q", axis=alt.Axis(format="%"))).properties(width=420, height=220)
    if {"feature", "mean_abs_shap"} <= set(r.columns):
        return alt.Chart(r).mark_bar().encode(y=alt.Y("feature:N", sort="-x"), x="mean_abs_shap:Q", color="segment:N", yOffset="segment:N").properties(width=380, height=220)
    return None


def explorer(month_name: str, day_type: str, weather: str, trip: float):
    m = MONTH_NAMES.index(month_name) + 1
    con = connect(read_only=True)
    df = con.execute("SELECT * FROM demand_cube WHERE month = ? AND calendar_class = ? AND weather_band = ? AND trip_median = ? ORDER BY hour",
                     [m, day_type, weather, int(trip)]).df()
    con.close()
    rows = []
    for seg in ["casual", "registered", "total"]:
        rows.append(df[["hour"]].assign(segment=seg, p10=df[f"{seg}_p10"], p50=df[f"{seg}_p50"], p90=df[f"{seg}_p90"]))
    long = pd.concat(rows)
    band = alt.Chart(long).mark_area(opacity=0.25).encode(x="hour:O", y=alt.Y("p10:Q", title="rentals per hour"), y2="p90:Q", color="segment:N")
    line = alt.Chart(long).mark_line(point=True).encode(x="hour:O", y="p50:Q", color="segment:N")
    chart = (band + line).properties(width=620, height=300, title=f"{month_name} · {day_type} · {weather}: p10–p90 bands (2012 level)")
    return chart, df[["hour", "total_p10", "total_p50", "total_p90", "bikes_in_use_p95", "bikes_95_service"]]


def simulator(service: float, friction: float, trip: float, cu: float, co: float):
    from .tools import simulate_fleet

    r = simulate_fleet(service / 100, friction, trip, cu, co)
    u = r.pop("utilisation")
    u = u[u.calendar_class.isin(["workday", "weekend"])]
    fleet = r["required_fleet"]
    ch = alt.Chart(u).mark_line(point=True).encode(x="hour:O", y=alt.Y("mean_in_use:Q", title="mean bikes in use"), color="calendar_class:N")
    rule = alt.Chart(pd.DataFrame({"y": [fleet]})).mark_rule(color="#c0392b", strokeDash=[4, 4]).encode(y="y:Q")
    summary = (f"### Required fleet: **{fleet:,.0f} bikes**\n"
               f"- service level {r['service_level']:.0%}, friction {r['friction']}, trip median {r['trip_median_used']:.0f} min\n"
               f"- idle share of bike-hours: **{r['idle_share']:.0%}**\n"
               f"- poor vs good rebalancing (1.5 vs 1.2): {r['fleet_poor_rebalancing']:,.0f} vs {r['fleet_good_rebalancing']:,.0f} "
               f"→ **{r['bikes_saved_by_rebalancing']:,.0f} bikes** saved\n"
               f"- placeholder cost/day: ${r['cost_per_day']:,.0f} (Cu ${cu}, Co ${co})")
    return summary, (ch + rule).properties(width=620, height=280, title="Bikes in use vs fleet (red)")


def build_app():
    import gradio as gr

    desk = Desk()
    mode = f"provider: **{desk.provider}**" + ("" if desk.use_llm else " (offline rules mode — set a key in .env for the LLM)")

    def respond(message, history):
        a = desk.ask(message)
        history = (history or []) + [{"role": "user", "content": message}, {"role": "assistant", "content": a.text}]
        trace = f"**intent:** {a.intent} ({a.route_how})  \n"
        if a.sql:
            trace += f"**SQL:**\n```sql\n{a.sql}\n```\n"
        if a.tool_call:
            trace += f"**tool:** `{a.tool_call}`  \n"
        if a.sources:
            trace += "**sources:** " + "; ".join(a.sources) + "  \n"
        if a.assumptions:
            trace += "**assumptions:** " + " ".join(a.assumptions)
        return history, "", trace, a.rows.head(50) if len(a.rows) else pd.DataFrame(), answer_chart(a)

    with gr.Blocks(title="Fleet or Flow? — Operations desk") as app:
        gr.Markdown(f"# Fleet or Flow? — operations desk\nAnswers come from the pre-computed demand cube in DuckDB. {mode}")
        with gr.Tab("Ask the desk"):
            chat = gr.Chatbot(type="messages", height=360)
            box = gr.Textbox(placeholder="e.g. How many bikes should be available at 6 PM on a rainy weekday in November?", label="Question")
            trace = gr.Markdown()
            table = gr.Dataframe(label="Result rows")
            plot = gr.Plot(label="Chart")
            box.submit(respond, [box, chat], [chat, box, trace, table, plot])
            gr.Examples(["How many bikes should be available at 6 PM on a rainy weekday in November?",
                         "Which hours on a hot Saturday in July need the most bikes?",
                         "How much smaller is casual demand on wet days compared with dry ones?",
                         "What fleet do we need for a 95% service level if trips average 25 minutes?",
                         "How many bikes would better rebalancing save us?", "What happened on 29 October 2012?",
                         "Why do you treat windspeed zero as missing?", "What share of the fleet sits idle at 3 AM on a weekday?"], inputs=box)
        with gr.Tab("Demand explorer"):
            with gr.Row():
                m = gr.Dropdown(MONTH_NAMES, value="Jun", label="Month")
                d = gr.Dropdown(["workday", "weekend", "holiday", "bridge"], value="workday", label="Day type")
                w = gr.Dropdown(["dry-mild", "dry-hot", "dry-cold", "wet"], value="dry-mild", label="Weather")
                t = gr.Dropdown([12, 18, 25], value=18, label="Trip median (min)")
            go = gr.Button("Show")
            ep, et = gr.Plot(), gr.Dataframe()
            go.click(explorer, [m, d, w, t], [ep, et])
        with gr.Tab("Fleet simulator"):
            c = config()
            with gr.Row():
                s1 = gr.Slider(80, 99.5, value=95, step=0.5, label="Service level (%)")
                s2 = gr.Slider(1.0, 1.6, value=c["fleet"]["friction_default"], step=0.05, label="Spatial friction")
                s3 = gr.Dropdown([12, 18, 25], value=18, label="Trip median (min)")
            with gr.Row():
                s4 = gr.Number(value=c["fleet"]["cost"]["missed_ride_cu"], label="Cu: cost of a missed ride ($, placeholder)")
                s5 = gr.Number(value=c["fleet"]["cost"]["bike_daily_co"], label="Co: bike cost per day ($, placeholder)")
            sb = gr.Button("Simulate")
            sm, sp = gr.Markdown(), gr.Plot()
            sb.click(simulator, [s1, s2, s3, s4, s5], [sm, sp])
    return app


if __name__ == "__main__":
    build_app().launch()
