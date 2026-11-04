"""Workstream 1 — profile facts, contract run, audit checks and repairs.

Outputs (warehouse): raw_hour, raw_day, spine, hour_clean, day_clean
Outputs (artifacts/tables): profile_facts.csv, contract_violations.csv, contract_violations_summary.csv,
    gap_runs.csv, season_month_crosstab.csv, workingday_logic.csv, redundancy.csv, disruption_days.csv,
    audit_summary.json, repaired_contract_violations.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import contract
from .common import ROOT, config, connect, log, save_csv, save_json, write_table


class ProfileFactError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------------- load
def load_raw(con) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = ROOT / config()["paths"]["raw"]
    con.execute(f"CREATE OR REPLACE TABLE raw_hour AS SELECT * FROM read_csv_auto('{raw / 'hour.csv'}', header=true)")
    con.execute(f"CREATE OR REPLACE TABLE raw_day AS SELECT * FROM read_csv_auto('{raw / 'day.csv'}', header=true)")
    hour = con.execute("SELECT * FROM raw_hour ORDER BY instant").df()
    day = con.execute("SELECT * FROM raw_day ORDER BY instant").df()
    for df in (hour, day):
        df["dteday"] = pd.to_datetime(df["dteday"]).dt.strftime("%Y-%m-%d")
        for c in df.columns:
            if c not in ("dteday",) and df[c].dtype.kind in "iu":
                df[c] = df[c].astype("int64")
            elif c not in ("dteday",) and df[c].dtype.kind == "f":
                df[c] = df[c].astype("float64")
    return hour, day


# ------------------------------------------------------------------------------------ profile facts
def profile_facts(con, hour: pd.DataFrame, day: pd.DataFrame) -> pd.DataFrame:
    """Reproduce Part 2 of the blueprint exactly; every fact computed by SQL or pandas here."""
    q = con.execute
    facts = []

    def add(name, expected, observed):
        ok = (abs(float(expected) - float(observed)) < 0.005) if isinstance(expected, float) else expected == observed
        facts.append({"fact": name, "expected": expected, "observed": observed, "reproduced": bool(ok)})

    add("hour.csv rows", 17379, len(hour))
    add("hour.csv columns", 17, hour.shape[1])
    add("day.csv rows", 731, len(day))
    add("day.csv columns", 16, day.shape[1])
    n_days = q("SELECT COUNT(DISTINCT dteday) FROM raw_hour").fetchone()[0]
    add("possible hours (731 x 24)", 17544, int(n_days) * 24)
    absent = q("""
        WITH spine AS (
            SELECT d::DATE AS dteday, h AS hr
            FROM range(DATE '2011-01-01', DATE '2013-01-01', INTERVAL 1 DAY) t(d), range(24) s(h))
        SELECT COUNT(*) FROM spine s ANTI JOIN raw_hour r ON s.dteday = r.dteday::DATE AND s.hr = r.hr
    """).fetchone()[0]
    add("hours absent from hour.csv", 165, int(absent))
    add("null cells (hour.csv)", 0, int(hour.isna().sum().sum()))
    add("duplicate rows (hour.csv)", 0, int(hour.duplicated().sum()))
    add("rows where casual + registered != cnt", 0, int((hour.casual + hour.registered != hour.cnt).sum()))
    mismatch = q("""
        SELECT COUNT(*) FROM raw_day d JOIN (
            SELECT dteday, SUM(casual) c, SUM(registered) r, SUM(cnt) t FROM raw_hour GROUP BY dteday) h
        ON d.dteday = h.dteday
        WHERE d.casual != h.c OR d.registered != h.r OR d.cnt != h.t
    """).fetchone()[0]
    add("days where hourly sums != day.csv", 0, int(mismatch))
    add("windspeed = 0 rows", 2180, int((hour.windspeed == 0).sum()))
    add("hum = 0 rows", 22, int((hour.hum == 0).sum()))
    add("weathersit = 4 rows", 3, int((hour.weathersit == 4).sum()))
    s1 = hour[hour.season == 1]
    s1_months = sorted(s1.mnth.unique().tolist())
    last_march = s1[s1.mnth == 3].dteday.max()
    add("season 1 months (Jan-mid Mar, plus late Dec)", "[1, 2, 3, 12]", str(s1_months))
    add("season 1 last March day", "2011/2012-03-20", "2011/2012-03-" + last_march[-2:])
    add("corr(temp, atemp) ~ 0.99", 0.99, round(float(hour[["temp", "atemp"]].corr().iloc[0, 1]), 2))
    yr = hour.groupby("yr").cnt.sum()
    add("rentals 2011", 1243103, int(yr.loc[0]))
    add("rentals 2012", 2049576, int(yr.loc[1]))
    pk = hour.loc[hour.cnt.idxmax()]
    add("peak hour", "2012-09-12 18:00 = 977", f"{pk.dteday} {int(pk.hr):02d}:00 = {int(pk.cnt)}")
    lo = day.loc[day.cnt.idxmin()]
    add("lowest day", "2012-10-29 = 22", f"{lo.dteday} = {int(lo.cnt)}")
    df = pd.DataFrame(facts)
    df["expected"] = df["expected"].astype(str)
    df["observed"] = df["observed"].astype(str)
    save_csv(df, "profile_facts")
    bad = df[~df.reproduced]
    if len(bad):
        raise ProfileFactError("Profile facts did not reproduce:\n" + bad.to_string())
    log.info("Profile facts: %d/%d reproduced", int(df.reproduced.sum()), len(df))
    return df


# -------------------------------------------------------------------------------------------- gaps
def classify_gaps(con) -> pd.DataFrame:
    """Anti-join against the full 17,544-hour spine; classify missing runs as night lull or outage."""
    cfg = config()["data"]
    night = cfg["night_lull_hours"]
    spine = con.execute("""
        WITH spine AS (
            SELECT d::DATE AS dteday, h AS hr
            FROM range(DATE '2011-01-01', DATE '2013-01-01', INTERVAL 1 DAY) t(d), range(24) s(h))
        SELECT s.dteday, s.hr, r.cnt IS NULL AS missing
        FROM spine s LEFT JOIN raw_hour r ON s.dteday = r.dteday::DATE AND s.hr = r.hr
        ORDER BY s.dteday, s.hr
    """).df()
    spine["ts"] = pd.to_datetime(spine.dteday) + pd.to_timedelta(spine.hr, unit="h")
    spine["run_id"] = (spine.missing != spine.missing.shift()).cumsum()
    runs = (
        spine[spine.missing]
        .groupby("run_id")
        .agg(start=("ts", "min"), end=("ts", "max"), n_hours=("ts", "size"), hr_min=("hr", "min"), hr_max=("hr", "max"))
        .reset_index()
    )
    runs["all_night"] = [
        all(((runs.start.iloc[i] + pd.Timedelta(hours=k)).hour in night) for k in range(runs.n_hours.iloc[i]))
        for i in range(len(runs))
    ]
    runs["gap_type"] = np.where(runs.all_night & (runs.n_hours <= cfg["night_lull_max_run"]), "night_lull", "outage")
    spine = spine.merge(runs[["run_id", "gap_type"]], on="run_id", how="left")
    spine["status"] = np.where(~spine.missing, "observed", spine.gap_type)
    spine = spine[["dteday", "hr", "status"]]
    spine["dteday"] = pd.to_datetime(spine.dteday).dt.strftime("%Y-%m-%d")
    runs_out = runs.drop(columns=["run_id", "all_night"])
    save_csv(runs_out, "gap_runs")
    write_table(con, "spine", spine)
    return spine


# -------------------------------------------------------------------------------------- repairs
def _impute_sentinel(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[f"{col}_imputed"] = (df[col] == 0).astype(int)
    df.loc[df[col] == 0, col] = np.nan
    df = df.sort_values(["dteday", "hr"])
    df[col] = df.groupby("dteday")[col].transform(lambda s: s.interpolate(limit_area="inside"))
    mh = df.groupby(["mnth", "hr"])[col].transform("median")
    df[col] = df[col].fillna(mh)
    return df


def repair(hour: pd.DataFrame, spine: pd.DataFrame) -> pd.DataFrame:
    cfg = config()["data"]
    df = hour.drop(columns=["instant"]).copy()
    df["imputed_row"] = 0
    # 1) night-lull gaps -> explicit zero-rental hours (assumption register; flagged)
    if cfg["night_gaps_as_zero"]:
        lulls = spine[spine.status == "night_lull"][["dteday", "hr"]]
        daycols = ["season", "yr", "mnth", "holiday", "weekday", "workingday"]
        dayinfo = df.groupby("dteday")[daycols].first().reset_index()
        add = lulls.merge(dayinfo, on="dteday", how="left")
        for c in ["casual", "registered", "cnt"]:
            add[c] = 0
        add["imputed_row"] = 1
        df = pd.concat([df, add], ignore_index=True).sort_values(["dteday", "hr"]).reset_index(drop=True)
        # weather for inserted rows: linear interpolation in time / carry for the category
        for c in ["temp", "atemp", "hum", "windspeed"]:
            df[c] = df[c].interpolate(limit_direction="both")
        df["weathersit"] = df["weathersit"].ffill().bfill()
    # 2) sentinel zeros -> null -> same-day interpolation -> month-hour median
    df = _impute_sentinel(df, "windspeed")
    df = _impute_sentinel(df, "hum")
    # 3) sparse category: fold weathersit 4 into 3
    df["weathersit_raw"] = df["weathersit"].astype(int)
    df["weathersit"] = df["weathersit"].clip(upper=3).astype(int)
    for c in ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "casual", "registered", "cnt"]:
        df[c] = df[c].astype("int64")
    df = df.sort_values(["dteday", "hr"]).reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------------- other checks
def label_semantics(hour: pd.DataFrame) -> dict:
    ct = pd.crosstab(hour.season, hour.mnth)
    ct.index = [f"season_{i}" for i in ct.index]
    save_csv(ct.reset_index().rename(columns={"index": "season"}), "season_month_crosstab")
    expected_wd = ((hour.weekday.between(1, 5)) & (hour.holiday == 0)).astype(int)
    wl = pd.crosstab([hour.holiday, hour.weekday.between(1, 5).rename("mon_fri")], hour.workingday)
    save_csv(wl.reset_index(), "workingday_logic")
    s1 = hour[hour.season == 1]
    return {
        "season1_months": sorted(s1.mnth.unique().tolist()),
        "season1_first_dec_day": s1[s1.mnth == 12].dteday.min(),
        "workingday_mismatch_rows": int((expected_wd != hour.workingday).sum()),
    }


def redundancy(df: pd.DataFrame) -> pd.DataFrame:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    cols = ["temp", "atemp", "hum", "windspeed"]
    X = df[cols].assign(const=1.0).to_numpy()
    vif = [variance_inflation_factor(X, i) for i in range(len(cols))]
    X2 = df[["temp", "hum", "windspeed"]].assign(const=1.0).to_numpy()
    vif2 = [variance_inflation_factor(X2, i) for i in range(3)] + [np.nan]
    out = pd.DataFrame({"variable": cols, "vif_all": vif, "vif_without_atemp": [vif2[0], np.nan, vif2[1], vif2[2]]})
    out["corr_with_temp"] = [df[c].corr(df["temp"]) for c in cols]
    save_csv(out, "redundancy")
    return out


def stl_daily(day_totals: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """STL on log daily totals (weekly period); robust z-score of the remainder flags disruptions."""
    from statsmodels.tsa.seasonal import STL

    z_thr = config()["data"]["disruption_z"]
    s = day_totals.set_index(pd.to_datetime(day_totals.dteday))["cnt"].asfreq("D")
    y = np.log(s.clip(lower=1))
    res = STL(y, period=7, robust=False).fit()  # robust=True shrinks the MAD and flags ~150 ordinary rainy days
    out = pd.DataFrame(
        {"dteday": s.index.strftime("%Y-%m-%d"), "cnt": s.values, "log_cnt": y.values,
         "stl_trend": res.trend.values, "stl_weekly": res.seasonal.values, "stl_remainder": res.resid.values}
    )
    r = out.stl_remainder
    mad = np.median(np.abs(r - np.median(r)))
    out["robust_z"] = (r - np.median(r)) / (1.4826 * mad)
    out["disruption_day"] = (out.robust_z.abs() > z_thr).astype(int)
    out["stl_trend_level"] = np.exp(out.stl_trend)
    # trend growth: slope of the log trend per month (30.44 days)
    t = np.arange(len(out))
    slope = np.polyfit(t, out.stl_trend.values, 1)[0] * 30.44
    wk = out.assign(dow=pd.to_datetime(out.dteday).dt.day_name()).groupby("dow").stl_weekly.mean()
    stats = {
        "trend_growth_pct_per_month": float((np.exp(slope) - 1) * 100),
        "trend_level_start": float(out.stl_trend_level.iloc[:30].mean()),
        "trend_level_end": float(out.stl_trend_level.iloc[-30:].mean()),
        "weekly_cycle_amplitude_pct": float((np.exp(wk.max() - wk.min()) - 1) * 100),
        "weekly_cycle_by_day_pct": {k: float((np.exp(v) - 1) * 100) for k, v in wk.items()},
        "remainder_sd_log": float(r.std()),
        "n_disruption_days": int(out.disruption_day.sum()),
    }
    return out, stats


# --------------------------------------------------------------------------------------- run
def run() -> dict:
    con = connect()
    try:
        hour, day = load_raw(con)
        facts = profile_facts(con, hour, day)

        viol = contract.validate(hour, contract.raw_schema())
        save_csv(viol, "contract_violations")
        vsum = (viol.groupby(["column", "check"], dropna=False).size().reset_index(name="n_failures")
                if len(viol) else pd.DataFrame(columns=["column", "check", "n_failures"]))
        save_csv(vsum, "contract_violations_summary")
        log.info("Raw contract: %d failure cases (expected: sentinel zeros)", len(viol))

        spine = classify_gaps(con)
        clean = repair(hour, spine)
        rviol = contract.validate(clean, contract.repaired_schema())
        save_csv(rviol, "repaired_contract_violations")
        if len(rviol):
            raise RuntimeError(f"Repaired data fails the contract ({len(rviol)} cases)")
        log.info("Repaired data passes the contract")

        sem = label_semantics(hour)
        red = redundancy(clean)

        daily = clean.groupby("dteday")[["casual", "registered", "cnt"]].sum().reset_index()
        cross_ok = bool((daily.merge(day, on="dteday", suffixes=("", "_d"))
                         .eval("cnt == cnt_d and casual == casual_d and registered == registered_d")).all())
        if not cross_ok:
            raise RuntimeError("Hourly sums (after repair) no longer match day.csv")
        stl, stl_stats = stl_daily(daily)
        day_clean = day.drop(columns=["instant"]).merge(
            stl[["dteday", "stl_trend_level", "stl_weekly", "stl_remainder", "robust_z", "disruption_day"]], on="dteday")
        day_clean["weathersit"] = day_clean["weathersit"].clip(upper=3)
        dd = day_clean[day_clean.disruption_day == 1][["dteday", "cnt", "weathersit", "holiday", "workingday", "robust_z"]]
        save_csv(dd, "disruption_days")

        clean = clean.merge(day_clean[["dteday", "disruption_day"]], on="dteday", how="left")
        write_table(con, "hour_clean", clean)
        write_table(con, "day_clean", day_clean)
        write_table(con, "stl_daily", stl)

        status = spine.status.value_counts().to_dict()
        summary = {
            "facts_reproduced": int(facts.reproduced.sum()), "facts_total": len(facts),
            "raw_contract_failures": int(len(viol)),
            "raw_contract_failures_by_column": vsum.groupby("column").n_failures.sum().to_dict() if len(vsum) else {},
            "gap_hours_night_lull": int(status.get("night_lull", 0)),
            "gap_hours_outage": int(status.get("outage", 0)),
            "gap_runs_outage": int((pd.read_csv(ROOT / config()["paths"]["tables"] / "gap_runs.csv").gap_type == "outage").sum()),
            "rows_inserted_zero": int(clean.imputed_row.sum()),
            "windspeed_imputed": int((hour.windspeed == 0).sum()),
            "hum_imputed": int((hour.hum == 0).sum()),
            "imputed_incl_inserted_rows": {"windspeed": int(clean.windspeed_imputed.sum()), "hum": int(clean.hum_imputed.sum())},
            "hum_zero_dates": sorted(hour[hour.hum == 0].dteday.unique().tolist()),
            "weathersit4_folded": int((clean.weathersit_raw == 4).sum()),
            "season_semantics": sem,
            "corr_temp_atemp": float(hour[["temp", "atemp"]].corr().iloc[0, 1]),
            "vif_temp_with_atemp": float(red.loc[red.variable == "temp", "vif_all"].iloc[0]),
            "vif_temp_without_atemp": float(red.loc[red.variable == "temp", "vif_without_atemp"].iloc[0]),
            "cross_file_equal": cross_ok,
            "stl": stl_stats,
            "disruption_days": dd.dteday.tolist(),
            "clean_rows": int(len(clean)),
        }
        save_json(summary, "audit_summary")
        return summary
    finally:
        con.close()
