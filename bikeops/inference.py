"""Workstream 2 — who stays home when the weather turns.

H1: wet weather reduces casual rentals by a substantially larger proportion than registered rentals
(ratio of rate-ratios casual/registered < 0.75). Two methods: NB GLM with date-clustered SEs and a
matched wet-vs-dry comparison. Plus temperature elasticity, robustness, supply-cap probe and the
power calculation for the capacity-elasticity field experiment.
"""
from __future__ import annotations

import gc
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from .common import ROOT, config, log, path, save_chart, save_csv, save_json, seed

CTRL = "is_casual*C(hour)*C(calendar_class) + C(month) + growth_index"


# ---------------------------------------------------------------------------------- pre-registration
def preregister() -> str:
    """Write the decision rule BEFORE any estimate is produced (kept if it already exists)."""
    p = path("docs") / "hypothesis_preregistration.md"
    if p.exists():
        return p.read_text()
    thr = config()["inference"]["ratio_threshold"]
    txt = f"""# Pre-registration — Workstream 2

Written by `bikeops.inference.preregister()` on {datetime.now():%Y-%m-%d %H:%M}, before any model in
Workstream 2 was estimated.

**H1.** Holding hour, calendar class, month and trend fixed, wet weather (weathersit 3, with 4 folded in)
reduces casual rentals by a substantially larger proportion than registered rentals.

**H0.** The wet-weather rate-ratio is the same for both segments.

**Test statistic.** Ratio of rate-ratios, RRR = RR(casual, wet vs dry) / RR(registered, wet vs dry).

**Decision rule.** Reject H0 in favour of H1 only if BOTH
1. the Negative Binomial GLM (date-clustered standard errors) gives RRR < {thr} with a 95% CI excluding 1, and
2. the matched wet-vs-dry comparison gives RRR < {thr} with a 95% bootstrap CI (over days) excluding 1.

If only one method meets the rule, the result is reported as *inconclusive*; if neither, H0 is not rejected.
Robustness checks (each year separately, excluding disruption days, mist as a second treatment) are
reported but do not change the decision.
"""
    p.write_text(txt)
    return txt


# ---------------------------------------------------------------------------------- data
def long_format(sig: pd.DataFrame) -> pd.DataFrame:
    obs = sig[sig.imputed_row == 0].copy()
    obs["mist"] = (obs.weathersit == 2).astype(int)
    keep = ["dteday", "date", "hour", "calendar_class", "month", "yr", "growth_index", "wet", "mist", "weathersit",
            "temp_c", "disruption_day"]
    a = obs[keep + ["casual"]].rename(columns={"casual": "rentals"}).assign(segment="casual", is_casual=1)
    b = obs[keep + ["registered"]].rename(columns={"registered": "rentals"}).assign(segment="registered", is_casual=0)
    return pd.concat([a, b], ignore_index=True)


def nb_alpha(y, mu) -> float:
    """Cameron–Trivedi auxiliary regression: ((y-mu)^2 - y)/mu = alpha*mu + e."""
    lhs = ((y - mu) ** 2 - y) / mu
    return float(max(np.sum(lhs * mu) / np.sum(mu * mu), 1e-4))


def fit_nb(formula: str, df: pd.DataFrame):
    pois = smf.glm(formula, df, family=sm.families.Poisson()).fit()
    alpha = nb_alpha(df.rentals.to_numpy(), pois.fittedvalues.to_numpy())
    del pois
    gc.collect()
    groups = pd.factorize(df.dteday)[0]
    res = smf.glm(formula, df, family=sm.families.NegativeBinomial(alpha=alpha)).fit(
        cov_type="cluster", cov_kwds={"groups": groups})
    res.nb_alpha = alpha
    return res


def model_rrr(df: pd.DataFrame, treat: str = "wet") -> dict:
    ctrl = CTRL if df.yr.nunique() > 1 else CTRL.replace(" + growth_index", "")  # within one year trend = month
    f = f"rentals ~ is_casual*{treat} + is_casual*temp_c + {ctrl}"
    res = fit_nb(f, df)
    name = f"is_casual:{treat}"
    b, se = res.params[name], res.bse[name]
    rr_reg = np.exp(res.params[treat])
    rr_cas = np.exp(res.params[treat] + b)
    return {"method": "NB GLM", "rrr": float(np.exp(b)), "ci_low": float(np.exp(b - 1.96 * se)),
            "ci_high": float(np.exp(b + 1.96 * se)), "p_value": float(res.pvalues[name]),
            "rr_casual": float(rr_cas), "rr_registered": float(rr_reg), "nb_alpha": res.nb_alpha,
            "n_obs": int(res.nobs), "_res": res}


# ---------------------------------------------------------------------------------- matched
def matched_rrr(sig: pd.DataFrame, treat: str = "wet", subset=None, reps: int | None = None) -> tuple[dict, pd.DataFrame]:
    cfg = config()["inference"]
    k = cfg["matched_neighbours"]
    reps = reps or cfg["bootstrap_reps"]
    obs = sig[sig.imputed_row == 0].copy()
    if subset is not None:
        obs = obs[subset(obs)]
    if treat == "wet":
        tr, ct = obs[obs.wet == 1], obs[obs.wet == 0]
    else:  # mist vs clear, wet hours excluded
        tr, ct = obs[obs.weathersit == 2], obs[obs.weathersit == 1]
    keys = ["hour", "calendar_class", "month", "yr"]
    m = tr[keys + ["dteday", "date", "casual", "registered"]].merge(
        ct[keys + ["dteday", "date", "casual", "registered"]], on=keys, suffixes=("_w", "_d"))
    m = m[m.dteday_w != m.dteday_d]
    m["gap"] = (m.date_w - m.date_d).abs().dt.days
    m = m.sort_values(["dteday_w", "hour", "gap"])
    m = m.groupby(["dteday_w", "hour"]).head(k)
    pairs = m.groupby(["dteday_w", "hour"]).agg(
        cas_w=("casual_w", "first"), reg_w=("registered_w", "first"),
        cas_d=("casual_d", "mean"), reg_d=("registered_d", "mean"), gap_days=("gap", "mean")).reset_index()
    pairs["d"] = (np.log((pairs.cas_w + 0.5) / (pairs.cas_d + 0.5)) - np.log((pairs.reg_w + 0.5) / (pairs.reg_d + 0.5)))
    byday = pairs.groupby("dteday_w")[["cas_w", "reg_w", "cas_d", "reg_d", "d"]].agg(
        {"cas_w": "sum", "reg_w": "sum", "cas_d": "sum", "reg_d": "sum", "d": "mean"})
    A = byday[["cas_w", "cas_d", "reg_w", "reg_d"]].to_numpy()

    def rrr(w):
        s = w @ A
        return (s[0] / s[1]) / (s[2] / s[3])

    est = rrr(np.ones(len(A)))
    rng = np.random.default_rng(seed())
    W = rng.multinomial(len(A), np.ones(len(A)) / len(A), size=reps)
    S = W @ A
    boot = (S[:, 0] / S[:, 1]) / (S[:, 2] / S[:, 3])
    wil = stats.wilcoxon(byday.d)
    out = {"method": "Matched wet vs dry", "rrr": float(est), "ci_low": float(np.quantile(boot, 0.025)),
           "ci_high": float(np.quantile(boot, 0.975)), "p_value": float(wil.pvalue),
           "rr_casual": float(A[:, 0].sum() / A[:, 1].sum()), "rr_registered": float(A[:, 2].sum() / A[:, 3].sum()),
           "median_pair_log_ratio_diff": float(np.median(pairs.d)), "n_pairs": int(len(pairs)), "n_days": int(len(byday)),
           "mean_gap_days": float(pairs.gap_days.mean())}
    return out, pairs


# ---------------------------------------------------------------------------------- elasticity
def temperature_elasticity(long: pd.DataFrame) -> pd.DataFrame:
    lo, hi = config()["inference"]["elasticity_band_c"]
    d = long[(long.temp_c >= lo) & (long.temp_c <= hi)]
    res = fit_nb(f"rentals ~ is_casual*temp_c + is_casual*wet + {CTRL}", d)
    res.remove_data()
    b_r, b_i = res.params["temp_c"], res.params["is_casual:temp_c"]
    V = res.cov_params()
    v_r = V.loc["temp_c", "temp_c"]
    v_c = v_r + V.loc["is_casual:temp_c", "is_casual:temp_c"] + 2 * V.loc["temp_c", "is_casual:temp_c"]
    rows = []
    for seg, b, v in [("registered", b_r, v_r), ("casual", b_r + b_i, v_c)]:
        se = np.sqrt(v)
        rows.append({"segment": seg, "pct_per_degC": (np.exp(b) - 1) * 100, "ci_low": (np.exp(b - 1.96 * se) - 1) * 100,
                     "ci_high": (np.exp(b + 1.96 * se) - 1) * 100, "band_c": f"{lo}-{hi}", "n_obs": int(res.nobs)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------- probe
def supply_cap_probe(sig: pd.DataFrame, res_main) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    p = config()["inference"]["probe"]
    obs = sig[sig.imputed_row == 0].copy()
    good = obs[(obs.calendar_class == "workday") & obs.hour.isin(p["hours"]) & (obs.weathersit == 1)
               & obs.temp_c.between(*p["temp_c"]) & (obs.hum_pct < p["hum_max"])].copy()
    good["ym"] = good.ts.dt.to_period("M").astype(str)
    monthly = good.groupby("ym").cnt.agg(n="size", mean="mean", p95=lambda s: s.quantile(0.95), max="max").reset_index()
    monthly["max_over_p95"] = monthly["max"] / monthly.p95
    monthly["rolling_p95_3m"] = monthly.p95.rolling(3, min_periods=1).mean()
    # predicted demand from the WS2 NB model (sum of the two segments)
    lg = long_format(sig)
    lg["mu"] = res_main.predict(lg)
    mu = lg.pivot_table(index=["dteday", "hour"], columns="segment", values="mu").sum(axis=1).rename("mu").reset_index()
    g = good.merge(mu, on=["dteday", "hour"])
    g["resid"] = g.cnt - g.mu
    g["bin"] = pd.qcut(g.mu, 5, labels=[f"Q{i}" for i in range(1, 6)])
    trunc = g.groupby("bin", observed=True).agg(n=("cnt", "size"), mean_pred=("mu", "mean"), mean_obs=("cnt", "mean"),
                                               resid_mean=("resid", "mean"), resid_sd=("resid", "std")).reset_index()
    trunc["resid_cv"] = trunc.resid_sd / trunc.mean_pred
    trunc["obs_over_pred"] = trunc.mean_obs / trunc.mean_pred
    slope = np.polyfit(np.arange(5), trunc.resid_cv.to_numpy(), 1)[0]
    info = {"n_hours": int(len(good)), "months": int(len(monthly)),
            "median_max_over_p95": float(monthly.max_over_p95.median()),
            "resid_cv_slope_per_quintile": float(slope),
            "top_quintile_obs_over_pred": float(trunc.obs_over_pred.iloc[-1]),
            "bottom_quintile_obs_over_pred": float(trunc.obs_over_pred.iloc[0]),
            "truncation_signal": bool(slope < 0 and trunc.obs_over_pred.iloc[-1] < trunc.obs_over_pred.iloc[0])}
    return monthly, trunc, info


# ---------------------------------------------------------------------------------- experiment power
def experiment_power(sig: pd.DataFrame, alpha_nb: float) -> dict:
    e = config()["inference"]["experiment"]
    obs = sig[(sig.imputed_row == 0) & (sig.yr == 1) & (sig.calendar_class == "workday") & sig.hour.isin([7, 8, 9])]
    mu_sys = float(obs.cnt.mean())
    # hourly variance estimated here: dispersion of AM-peak workday hours around their hour x month mean
    g = obs.groupby(["hour", "month"]).cnt
    resid = obs.cnt - g.transform("mean")
    var_sys = float(resid.var())
    alpha_hat = max((var_sys - mu_sys) / mu_sys**2, 0.0)
    mu_c = mu_sys / e["clusters"]
    hours_per_week = 15  # 5 workdays x 3 AM-peak hours
    cv2_hour = 1 / mu_c + alpha_hat           # Poisson + extra (weather/day) variation per cluster-hour
    cv2_week = cv2_hour / hours_per_week + e["cluster_noise_cv"] ** 2
    n_arm = e["clusters"] * e["weeks"] / e["arms"]
    z = stats.norm.ppf(1 - e["alpha"] / 2) + stats.norm.ppf(e["power"])
    mde_log = z * np.sqrt(2 * cv2_week / n_arm)
    mde_pct = (np.exp(mde_log) - 1) * 100
    # simulation check of power for a 5% and a 10% effect
    rng = np.random.default_rng(seed())
    sd = np.sqrt(cv2_week)

    def sim_power(effect, reps=4000):
        n = int(n_arm)
        a = rng.normal(0, sd, (reps, n))
        b = rng.normal(np.log(1 - effect), sd, (reps, n))
        t = (a.mean(1) - b.mean(1)) / np.sqrt(a.var(1, ddof=1) / n + b.var(1, ddof=1) / n)
        return float((np.abs(t) > stats.norm.ppf(1 - e["alpha"] / 2)).mean())

    return {"design": "switchback, 8 weeks, 3 arms (normal / -10% withheld / +10% pre-placed)",
            "clusters": e["clusters"], "cluster_weeks_per_arm": n_arm, "am_peak_mean_system_rentals_per_hour": mu_sys,
            "am_peak_hourly_variance": var_sys, "extra_dispersion_alpha": alpha_hat, "nb_alpha_ws2_model": alpha_nb,
            "mean_rentals_per_cluster_hour": mu_c, "cv_cluster_week": float(sd),
            "assumed_cluster_noise_cv": e["cluster_noise_cv"], "mde_pct": float(mde_pct),
            "power_at_5pct": sim_power(0.05), "power_at_10pct": sim_power(0.10)}


# ---------------------------------------------------------------------------------- figures
def fig_forest(tab: pd.DataFrame):
    thr = config()["inference"]["ratio_threshold"]
    d = tab.assign(label=tab.analysis + " · " + tab.method)
    base = alt.Chart(d).encode(y=alt.Y("label:N", sort=None, title=None))
    bars = base.mark_rule(strokeWidth=2).encode(x=alt.X("ci_low:Q", title="ratio of rate-ratios (casual / registered)", scale=alt.Scale(domain=[0, 1.2])), x2="ci_high:Q", color=alt.Color("method:N"))
    pts = base.mark_point(filled=True, size=70).encode(x="rrr:Q", color="method:N")
    r1 = alt.Chart(pd.DataFrame({"x": [1.0]})).mark_rule(strokeDash=[4, 4], color="#555").encode(x="x:Q")
    r2 = alt.Chart(pd.DataFrame({"x": [thr]})).mark_rule(color="#c0392b").encode(x="x:Q")
    return (bars + pts + r1 + r2).properties(width=480, height=24 * len(d), title=f"Weather sensitivity: casual vs registered (red line = {thr} rule)")


def fig_probe(monthly: pd.DataFrame):
    long = monthly.melt(id_vars="ym", value_vars=["mean", "p95", "max"], var_name="stat", value_name="rentals")
    return alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("ym:O", title="month"), y=alt.Y("rentals:Q", title="rentals per hour"), color="stat:N",
    ).properties(width=560, height=220, title="Supply-cap probe: best-condition commute hours (exploratory)")


# ---------------------------------------------------------------------------------- run
def run(sig: pd.DataFrame) -> dict:
    preregister()
    thr = config()["inference"]["ratio_threshold"]
    long = long_format(sig)
    log.info("WS2: NB GLM main model on %d rows", len(long))
    main = model_rrr(long, "wet")
    res_main = main.pop("_res")
    mt, pairs = matched_rrr(sig, "wet")
    save_csv(pairs, "matched_pairs")
    rows = [dict(analysis="Main (2011-12, wet)", **main), dict(analysis="Main (2011-12, wet)", **mt)]

    rob = [("2011 only", lambda d: d.yr == 0, "wet"), ("2012 only", lambda d: d.yr == 1, "wet"),
           ("Excl. disruption days", lambda d: d.disruption_day == 0, "wet"), ("Mist as treatment", None, "mist")]
    for name, sub, treat in rob:
        lg = long if sub is None else long[sub(long)]
        if treat == "mist":
            lg = lg[lg.wet == 0]
        m = model_rrr(lg, treat)
        m.pop("_res")
        gc.collect()
        mm, _ = matched_rrr(sig, treat, subset=sub, reps=1000)
        rows += [dict(analysis=name, **m), dict(analysis=name, **mm)]
    tab = pd.DataFrame(rows)
    save_csv(tab, "hypothesis_results")
    save_chart(fig_forest(tab), "ws2_rrr_forest")

    meets = lambda r: (r["rrr"] < thr) and (r["ci_high"] < 1)  # noqa: E731
    m_ok, t_ok = meets(main), meets(mt)
    decision = "reject H0 (H1 supported)" if (m_ok and t_ok) else ("inconclusive" if (m_ok or t_ok) else "do not reject H0")

    el = temperature_elasticity(long)
    save_csv(el, "temperature_elasticity")
    monthly, trunc, pinfo = supply_cap_probe(sig, res_main)
    save_csv(monthly, "supply_cap_probe_monthly")
    save_csv(trunc, "supply_cap_probe_truncation")
    save_chart(fig_probe(monthly), "ws2_supply_cap_probe")
    power = experiment_power(sig, main["nb_alpha"])
    save_json(power, "experiment_power")

    summary = {"decision": decision, "threshold": thr, "model": main, "matched": mt,
               "model_meets_rule": bool(m_ok), "matched_meets_rule": bool(t_ok),
               "elasticity": el.to_dict(orient="records"), "probe": pinfo, "power": power,
               "prereg_file": str((path("docs") / "hypothesis_preregistration.md").relative_to(ROOT))}
    save_json(summary, "ws2_summary")
    return summary
