"""Scenario grid + concurrency runs (WS3) and the demand cube / desk tables (WS4.1).

demand_cube grain: month x calendar_class x hour x weather_band x trip_median, at the 2012 demand level.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import archetypes as arch
from . import concurrency as cc
from . import forecast as fc
from . import signals as S
from .common import ROOT, config, connect, log, save_csv, seed, write_table
from .fleet import FleetModel

CLASSES = ["workday", "weekend", "holiday", "bridge"]
BANDS = ["dry-mild", "dry-hot", "dry-cold", "wet"]
REP_DOW = {"workday": 2, "weekend": 5, "holiday": 0, "bridge": 4}  # Wed, Sat, Mon, Fri


# ---------------------------------------------------------------------------------- day mix
def day_bands(sig: pd.DataFrame) -> pd.DataFrame:
    b = config()["signals"]["weather_bands"]
    day = sig[sig.hour.between(7, 20)].groupby("dteday").agg(
        month=("month", "first"), calendar_class=("calendar_class", "first"), wet_hours=("wet", "sum"),
        tmax=("temp_c", "max"), tmean=("temp_c", "mean")).reset_index()
    day["weather_band"] = np.where(day.wet_hours >= 3, "wet", np.where(day.tmax >= b["hot_from_c"], "dry-hot",
                                   np.where(day.tmean < b["cold_below_c"], "dry-cold", "dry-mild")))
    return day


def day_mix(sig: pd.DataFrame) -> pd.DataFrame:
    return day_bands(sig).groupby(["month", "calendar_class", "weather_band"]).size().rename("n_days").reset_index()


# ---------------------------------------------------------------------------------- scenarios
def scenario_frame(sig: pd.DataFrame, prior: dict) -> pd.DataFrame:
    obs = sig[sig.imputed_row == 0]
    mb = obs.groupby(["month", "weather_band"]).agg(t=("temp_c", "median"), hum=("hum_pct", "median"), wind=("wind_kmh", "median"),
                                                    support=("temp_c", "size"))
    m_med = obs.groupby("month").agg(t=("temp_c", "median"), hum=("hum_pct", "median"), wind=("wind_kmh", "median"))
    mh = obs.groupby(["month", "hour"]).temp_c.median()
    fb = {"dry-hot": 29.5, "dry-cold": 7.0}
    rows = []
    for m in range(1, 13):
        for b in BANDS:
            if (m, b) in mb.index and mb.loc[(m, b), "support"] >= 6:
                t0, hum, wind, sup = mb.loc[(m, b), ["t", "hum", "wind", "support"]]
            else:
                sup = mb.loc[(m, b), "support"] if (m, b) in mb.index else 0
                t0 = fb.get(b, float(np.clip(m_med.loc[m, "t"], 12, 26)))
                hum, wind = m_med.loc[m, "hum"], m_med.loc[m, "wind"]
            for c in CLASSES:
                for h in range(24):
                    t = t0 + (mh.loc[(m, h)] - m_med.loc[m, "t"])
                    rows.append({"month": m, "weather_band": b, "calendar_class": c, "hour": h, "temp_c": float(t),
                                 "hum_pct": float(hum), "wind_kmh": float(wind), "weather_support_hours": int(sup)})
    sc = pd.DataFrame(rows)
    sc["dow"] = sc.calendar_class.map(REP_DOW)
    sc["holiday"] = (sc.calendar_class == "holiday").astype(int)
    sc["workingday"] = sc.calendar_class.isin(["workday", "bridge"]).astype(int)
    sc["wet"] = (sc.weather_band == "wet").astype(int)
    sc["weather_cat"] = np.where(sc.wet == 1, 3, 1)
    sc["met_season"] = sc.month.map(S.MET_SEASON)
    sc["comfort_index"] = S.comfort_index(sc.temp_c, sc.hum_pct, sc.wind_kmh)
    sc["date"] = pd.to_datetime(dict(year=2012, month=sc.month, day=15))
    sc["daylight"] = S.daylight_flags(sc["date"], sc["hour"])
    sc["growth_index"] = float(sig.growth_index.max())  # frozen at the 2012 level
    sc["archetype_prior"] = arch.apply_prior(sc, prior).to_numpy()
    return sc


def predict_scenarios(sc: pd.DataFrame, final: dict, bp: pd.DataFrame, champion: str, mondrian: bool) -> pd.DataFrame:
    X = S.model_matrix(sc)
    out = sc.copy()
    for seg in fc.SEGMENTS:
        out[f"{seg}_p50"] = fc.catboost_predict(final["models"][seg], X)
    out["total_p50"] = out.casual_p50 + out.registered_p50
    cal = bp[(bp.contender == champion) & (bp.imputed_row == 0)].copy()
    cg = fc.hour_group(cal) if mondrian else None
    tg = fc.hour_group(out) if mondrian else None
    for seg, col in [("casual", "casual"), ("registered", "registered"), ("total", "cnt")]:
        iv = fc.conformal_intervals(cal[f"pred_{col}"], cal[col], out[f"{seg}_p50"], (0.8,), cg, tg)
        out[f"{seg}_p10"], out[f"{seg}_p90"] = iv[0.8]
    return out


# ---------------------------------------------------------------------------------- concurrency
def run_concurrency(sig: pd.DataFrame, scen: pd.DataFrame, rq_tab: pd.DataFrame, rho: float) -> FleetModel:
    trips = config()["concurrency"]["trip_median_grid"]
    t = rq_tab[rq_tab.target == "cnt"]
    rq = {g: (d.q.to_numpy(), d.r.to_numpy()) for g, d in t.groupby("hgroup")}
    peaks, hourly = [], []
    cells = scen.groupby(["month", "calendar_class", "weather_band"])
    for i, ((m, c, b), g) in enumerate(cells):
        g = g.sort_values("hour")
        res = cc.simulate_cell(g.total_p50.to_numpy(), fc.hour_group(g), rq, rho, trips, seed=seed() + i)
        for tm in trips:
            r = res[tm]
            peaks.append(pd.DataFrame({"month": m, "calendar_class": c, "weather_band": b, "trip_median": tm,
                                       "draw": np.arange(len(r["peak"])), "peak": r["peak"].astype(np.int32)}))
            hourly.append(pd.DataFrame({"month": m, "calendar_class": c, "weather_band": b, "trip_median": tm, "hour": range(24),
                                        "mean_in_use": r["hourly_mean"].mean(0), "p50_in_use": np.median(r["hourly_max"], 0),
                                        "p95_in_use": np.quantile(r["hourly_max"], 0.95, axis=0),
                                        "rentals_sim_mean": res["counts"].mean(0)}))
        if i % 24 == 0:
            log.info("concurrency cell %d/%d", i + 1, cells.ngroups)
    peaks, hourly = pd.concat(peaks, ignore_index=True), pd.concat(hourly, ignore_index=True)
    con = connect()
    write_table(con, "concurrency_peaks", peaks)
    write_table(con, "concurrency_hourly", hourly)
    weights = day_mix(sig)
    write_table(con, "day_mix", weights)
    write_table(con, "scenario_predictions", scen)
    con.close()
    pk = peaks.groupby(["month", "calendar_class", "weather_band", "trip_median"]).peak.quantile([0.5, 0.95]).unstack()
    pk.columns = ["peak_p50", "peak_p95"]
    save_csv(pk.reset_index(), "concurrency_peak_by_cell")
    return FleetModel(peaks, hourly, weights)


def load_fleet_model(con=None) -> FleetModel:
    own = con is None
    con = con or connect(read_only=True)
    fm = FleetModel(con.execute("SELECT * FROM concurrency_peaks").df(), con.execute("SELECT * FROM concurrency_hourly").df(),
                    con.execute("SELECT * FROM day_mix").df())
    if own:
        con.close()
    return fm


# ---------------------------------------------------------------------------------- notes
def chunk_notes() -> pd.DataFrame:
    rows = []
    files = sorted((ROOT / "docs").glob("*.md")) + [ROOT / config()["paths"]["raw"] / "Readme.txt"]
    for f in files:
        heading = f.stem
        buf = []
        text = f.read_text(errors="ignore")
        for line in text.splitlines() + [""]:
            if re.match(r"^#{1,6} ", line):
                if buf:
                    rows.append((f.name, heading, " ".join(buf)))
                    buf = []
                heading = line.lstrip("# ").strip()
            elif line.strip() == "":
                if buf:
                    rows.append((f.name, heading, " ".join(buf)))
                    buf = []
            else:
                buf.append(line.strip())
    df = pd.DataFrame(rows, columns=["source", "heading", "text"])
    df = df[df.text.str.len() > 25].reset_index(drop=True)
    df.insert(0, "note_id", np.arange(1, len(df) + 1))
    return df


def build_notes(con) -> int:
    notes = chunk_notes()
    write_table(con, "notes", notes)
    try:
        con.execute("INSTALL fts; LOAD fts;")
        con.execute("PRAGMA create_fts_index('notes', 'note_id', 'heading', 'text', overwrite=1)")
    except Exception as e:  # pragma: no cover - offline without the extension
        log.warning("DuckDB FTS unavailable (%s); desk will fall back to keyword scoring", e)
    return len(notes)


# ---------------------------------------------------------------------------------- materialise
def materialise() -> dict:
    con = connect()
    scen = con.execute("SELECT * FROM scenario_predictions").df()
    hourly = con.execute("SELECT * FROM concurrency_hourly").df()
    f0 = config()["fleet"]["friction_default"]
    cube = scen.merge(hourly, on=["month", "calendar_class", "weather_band", "hour"])
    cube["bikes_in_use_p50"] = cube.p50_in_use.round(1)
    cube["bikes_in_use_p95"] = cube.p95_in_use.round(1)
    cube["bikes_95_service"] = np.ceil(cube.p95_in_use * f0)
    cols = ["month", "calendar_class", "hour", "weather_band", "trip_median",
            "casual_p10", "casual_p50", "casual_p90", "registered_p10", "registered_p50", "registered_p90",
            "total_p10", "total_p50", "total_p90", "bikes_in_use_p50", "bikes_in_use_p95", "bikes_95_service",
            "temp_c", "weather_support_hours"]
    cube = cube[cols].copy()
    for c in cols[5:14]:
        cube[c] = cube[c].round(1)
    cube["temp_c"] = cube.temp_c.round(1)
    write_table(con, "demand_cube", cube)

    sig = con.execute("SELECT * FROM signals").df()
    lab = con.execute("SELECT * FROM day_archetypes").df()
    hh = sig.merge(lab[["dteday", "archetype"]], on="dteday", how="left")
    hh = hh.assign(weather=hh.weathersit.map({1: "clear", 2: "mist", 3: "wet"}))
    hist_h = hh[["date", "dteday", "hour", "month", "yr", "calendar_class", "weather", "weather_band", "wet", "temp_c", "hum_pct",
                 "wind_kmh", "casual", "registered", "cnt", "archetype", "imputed_row", "disruption_day"]].copy()
    hist_h["date"] = pd.to_datetime(hist_h.date).dt.date
    hist_h["year"] = hist_h.yr + 2011
    write_table(con, "history_hourly", hist_h.drop(columns=["yr", "dteday"]))
    day = con.execute("SELECT * FROM day_clean").df()
    bands = day_bands(sig)[["dteday", "weather_band"]]
    cls = sig.groupby("dteday").calendar_class.first().reset_index()
    hd = day.merge(lab[["dteday", "archetype"]], on="dteday", how="left").merge(bands, on="dteday", how="left").merge(cls, on="dteday", how="left")
    hd = hd.assign(date=pd.to_datetime(hd.dteday).dt.date, year=hd.yr + 2011, temp_c=hd.temp * config()["data"]["scale"]["temp"],
                   weather=hd.weathersit.map({1: "clear", 2: "mist", 3: "wet"}))
    hist_d = hd[["date", "year", "mnth", "weekday", "calendar_class", "weather", "weather_band", "temp_c", "casual", "registered",
                 "cnt", "archetype", "disruption_day", "robust_z", "stl_trend_level"]].rename(columns={"mnth": "month", "stl_trend_level": "stl_trend"})
    write_table(con, "history_daily", hist_d)

    fs = pd.read_csv(ROOT / config()["paths"]["tables"] / "fleet_service_level_grid.csv")
    write_table(con, "fleet_scenarios", fs)
    for name in ["fleet_utilisation", "fleet_vs_flow", "shap_summary", "hypothesis_results", "temperature_elasticity"]:
        write_table(con, name, pd.read_csv(ROOT / config()["paths"]["tables"] / f"{name}.csv"))
    n_notes = build_notes(con)
    out = {"cube_rows": len(cube), "history_hourly_rows": len(hist_h), "history_daily_rows": len(hist_d),
           "fleet_scenarios_rows": len(fs), "notes": n_notes}
    con.close()
    return out
