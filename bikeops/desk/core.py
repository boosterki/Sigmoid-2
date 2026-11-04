"""The desk: question -> router -> SQL agent / tools -> answer writer (numbers only from returned rows)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd

from ..common import config
from . import llm, router, sql_agent, tools
from .resolver import Slots, resolve

WRITER_SYSTEM = """You are the operations desk of a bike-share company. Answer the question using ONLY the rows/notes provided.
Quote numbers exactly as they appear (you may round to whole bikes or 1 decimal); never compute new numbers.
Name the table or document you used, and state assumptions (2012 demand level, trip median, friction, served-demand caveat).
At most 5 sentences. Return {"answer": "<text>"}."""


@dataclass
class Answer:
    question: str
    intent: str
    text: str
    sql: str | None = None
    tool_call: str | None = None
    rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    sources: list = field(default_factory=list)
    assumptions: list = field(default_factory=list)
    route_how: str = "rules"
    writer: str = "rules"
    error: str | None = None
    values: dict = field(default_factory=dict)  # key numbers for tests

    def chart(self):
        from .ui import answer_chart

        return answer_chart(self)


def _fmt(x, nd=0):
    try:
        return f"{float(x):,.{nd}f}"
    except Exception:
        return str(x)


class Desk:
    def __init__(self, provider: str | None = None):
        self.provider = llm.provider_name(provider)
        self.use_llm = self.provider != "rules" and llm.available(self.provider)

    # ------------------------------------------------------------------ main
    def ask(self, question: str) -> Answer:
        intent, how = router.route(question, self.provider if self.use_llm else "rules")
        slots = resolve(question)
        ans = Answer(question=question, intent=intent, text="", route_how=how)
        ans.assumptions += slots.notes
        try:
            if intent in ("lookup", "compare", "history"):
                self._sql(question, intent, slots, ans)
            elif intent == "what_if_fleet":
                self._fleet(question, slots, ans)
            else:
                self._explain(question, slots, ans)
        except Exception as e:
            ans.error = str(e)
            ans.text = f"I could not answer that from the demand cube ({e})."
            return ans
        if self.use_llm and not ans.error and len(ans.rows):
            try:
                payload = {"question": question, "intent": intent, "rows": ans.rows.head(30).to_dict(orient="records"),
                           "sources": ans.sources, "assumptions": ans.assumptions, "draft": ans.text}
                out = llm.complete(WRITER_SYSTEM, json.dumps(payload, default=str), self.provider)
                if out.get("answer"):
                    ans.text, ans.writer = out["answer"], "llm"
            except Exception:
                pass
        return ans

    # ------------------------------------------------------------------ intents
    def _sql(self, q: str, intent: str, slots: Slots, ans: Answer):
        if self.use_llm:
            r = sql_agent.llm_sql(q, slots, self.provider)
            if r.error:
                rs = sql_agent.rules_sql(q, intent, slots)
                if rs is None:
                    ans.error = r.error
                    ans.text = "I could not answer that: the query failed twice."
                    ans.sql = r.sql
                    return
                r = sql_agent.execute(rs)
        else:
            r = sql_agent.execute(sql_agent.rules_sql(q, intent, slots))
        ans.sql, ans.rows = r.sql, r.rows
        table = r.sql.split("FROM")[1].split()[0] if "FROM" in r.sql else "?"
        ans.sources.append(f"table {table}")
        if r.truncated:
            ans.assumptions.append(f"result truncated to {config()['desk']['max_rows']} rows")
        if table == "demand_cube":
            ans.assumptions.append(config()["desk"]["demand_level_note"] + " Forecast of served demand (true peaks likely higher).")
        if intent == "history" and slots.date:
            notes = tools.search_notes(f"{slots.date} disruption hurricane storm anomaly")
            if len(notes):
                ans.sources += [f"{n.source} › {n.heading}" for n in notes.itertuples()]
                ans.values["notes"] = notes.text.tolist()
        ans.text = self._template(q, intent, slots, ans)

    def _fleet(self, q: str, slots: Slots, ans: Answer):
        ql = q.lower()
        res = tools.simulate_fleet(slots.service_level, slots.friction, slots.trip_median)
        ans.tool_call = (f"simulate_fleet(service_level={res['service_level']}, friction={res['friction']}, "
                         f"trip_median={res['trip_median_used']})")
        ans.sources.append("tool simulate_fleet (concurrency_peaks, concurrency_hourly, day_mix)")
        util = res.pop("utilisation")
        ans.rows = pd.DataFrame([{k: v for k, v in res.items()}])
        ans.values = {**res, "utilisation": util}
        if res["trip_median_requested"] != res["trip_median_used"]:
            ans.assumptions.append(f"trip median {res['trip_median_requested']} min mapped to nearest simulated value {res['trip_median_used']} min")
        if "average" in ql or "mean" in ql:
            ans.assumptions.append("'average' trip length interpreted as the median of the trip-duration prior")
        ans.assumptions.append("service level = share of days on which the fleet covers the daily peak; costs are placeholders")
        if "rebalanc" in ql or "friction" in ql:
            ans.text = (f"Better rebalancing would save about {_fmt(res['bikes_saved_by_rebalancing'])} bikes: at a "
                        f"{res['service_level']:.0%} service level and {res['trip_median_used']:.0f}-minute median trips we need "
                        f"{_fmt(res['fleet_poor_rebalancing'])} bikes with poor rebalancing (friction 1.5) versus "
                        f"{_fmt(res['fleet_good_rebalancing'])} with good rebalancing (friction 1.2).")
        else:
            ans.text = (f"For a {res['service_level']:.0%} service level with {res['trip_median_used']:.0f}-minute median trips and "
                        f"spatial friction {res['friction']}, the fleet should be {_fmt(res['required_fleet'])} bikes. "
                        f"About {res['idle_share']:.0%} of bike-hours would be idle; placeholder cost ≈ ${_fmt(res['cost_per_day'])}/day.")

    def _explain(self, q: str, slots: Slots, ans: Answer):
        ql = q.lower()
        if "driver" in ql or "drives" in ql or "matter" in ql or "important" in ql:
            df = tools.explain_driver()
            ans.rows, ans.tool_call = df, "explain_driver()"
            ans.sources.append("table shap_summary")
            top = df[df["rank"] == 1]
            ans.text = "Top drivers by mean |SHAP|: " + "; ".join(f"{r.segment}: {r.feature}" for r in top.itertuples()) + "."
            return
        notes = tools.search_notes(q)
        ans.rows, ans.tool_call = notes, f"search_notes({q!r})"
        ans.sources += [f"{n.source} › {n.heading}" for n in notes.itertuples()]
        ans.values["notes"] = notes.text.tolist() if len(notes) else []
        ans.text = (notes.text.iloc[0] if len(notes) else "I found nothing in the project notes on that.")

    # ------------------------------------------------------------------ templated writer (offline)
    def _template(self, q: str, intent: str, slots: Slots, ans: Answer) -> str:
        r = ans.rows
        if r.empty:
            return "No matching rows."
        cols = set(r.columns)
        if "idle_share" in cols and "fleet" in cols:
            row = r.iloc[0]
            ans.values.update(idle_share=float(row.idle_share), fleet=float(row.fleet), hour=int(row.hour))
            return (f"At {int(row.hour):02d}:00 on a {slots.calendar_class or 'workday'}, about {row.idle_share:.0%} of the recommended "
                    f"fleet of {_fmt(row.fleet)} bikes sits idle (≈{_fmt(row.mean_in_use)} bikes on the road on average).")
        if "bikes_95_service" in cols:
            row = r.iloc[0]
            ans.values.update(bikes_95_service=float(row.bikes_95_service), hour=int(row.hour), total_p50=float(row.total_p50))
            desc = f"{slots.weather_band or 'dry-mild'} {slots.calendar_class or 'workday'} in month {slots.month or 6}"
            if slots.hour is not None:
                return (f"At {int(row.hour):02d}:00 on a {desc}, plan for {_fmt(row.bikes_95_service)} bikes available "
                        f"(95% service incl. friction 1.3). Expected rentals that hour: {_fmt(row.total_p50)} "
                        f"(80% band {_fmt(row.get('total_p10', float('nan')))}–{_fmt(row.total_p90)}).")
            ans.values["top_hours"] = r.hour.astype(int).tolist()
            hrs = ", ".join(f"{int(h):02d}:00 ({_fmt(b)} bikes)" for h, b in zip(r.hour, r.bikes_95_service, strict=True))
            return f"On a {desc}, the hours needing the most bikes are: {hrs}."
        if "weather_group" in cols:
            w = r.set_index("weather_group")
            pct = 1 - w.loc["wet", "avg_casual"] / w.loc["dry", "avg_casual"]
            pr = 1 - w.loc["wet", "avg_registered"] / w.loc["dry", "avg_registered"]
            ans.values.update(casual_pct_smaller=float(pct), registered_pct_smaller=float(pr))
            txt = (f"In wet hours casual rentals average {_fmt(w.loc['wet', 'avg_casual'], 1)} vs {_fmt(w.loc['dry', 'avg_casual'], 1)} "
                   f"in dry hours — {pct:.0%} smaller (members: {pr:.0%} smaller).")
            try:
                hr = sql_agent.execute("SELECT method, rr_casual, rr_registered, rrr, ci_low, ci_high FROM hypothesis_results "
                                       "WHERE analysis LIKE 'Main%'").rows
                m = hr[hr.method == "NB GLM"].iloc[0]
                ans.values.update(adjusted_rr_casual=float(m.rr_casual), rrr=float(m.rrr))
                txt += (f" Holding hour, day type, month and trend fixed (NB GLM), wet weather multiplies casual demand by "
                        f"{m.rr_casual:.2f} vs {m.rr_registered:.2f} for members (ratio {m.rrr:.2f}, 95% CI {m.ci_low:.2f}–{m.ci_high:.2f}).")
                ans.sources.append("table hypothesis_results")
            except Exception:
                pass
            return txt
        if "robust_z" in cols and len(r) == 1:
            row = r.iloc[0]
            ans.values.update(cnt=int(row.cnt), disruption_day=int(row.disruption_day))
            t = (f"On {row.date} the system recorded {_fmt(row.cnt)} rentals ({_fmt(row.casual)} casual, {_fmt(row.registered)} members) "
                 f"against a trend level of about {_fmt(row.stl_trend)} per day; weather '{row.weather}', robust z = {row.robust_z:.1f}"
                 f"{' — flagged as a disruption day' if row.disruption_day else ''}.")
            notes = " ".join(ans.values.get("notes", []))
            if "Sandy" in notes:
                t += " The project notes attribute this to Hurricane Sandy (system largely shut down)."
            return t
        return "Here are the rows: " + r.head(5).to_string(index=False)
