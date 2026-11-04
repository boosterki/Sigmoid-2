"""Workstream 3.1–3.3 — walk-forward probabilistic forecast of served demand.

Contenders: seasonal naive, profile x level, NB GLM, CatBoost Tweedie (champion candidate),
CatBoost + recent history (diagnostic, 1-day-ahead, not used by the desk).
Intervals: MAPIE split-conformal on walk-forward residuals (only earlier folds calibrate a fold),
Mondrian by hour group if peaks are under-covered.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import altair as alt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.base import BaseEstimator, RegressorMixin

from . import archetypes as arch
from . import signals as S
from .common import config, connect, log, path, save_chart, save_csv, save_json, seed, write_table

warnings.filterwarnings("ignore", category=FutureWarning)
SEGMENTS = ["casual", "registered"]
CONTENDERS = ["seasonal_naive", "profile_level", "nb_glm", "catboost_tweedie", "catboost_lags"]
SIMPLE = ["seasonal_naive", "profile_level", "nb_glm"]
DIAGNOSTIC = ["catboost_lags"]


# ------------------------------------------------------------------------------------------ folds
@dataclass
class Fold:
    idx: int
    start: pd.Timestamp
    end: pd.Timestamp
    kind: str  # warmup | tune | honest


def make_folds() -> list[Fold]:
    c = config()["forecast"]
    h = pd.Timedelta(days=c["horizon_days"])
    b0, bend = pd.Timestamp(c["backtest_start"]), pd.Timestamp(c["backtest_end"])
    out = []
    for w in range(c["warmup_folds"], 0, -1):
        s = b0 - w * h
        out.append(Fold(-w, s, s + h - pd.Timedelta(days=1), "warmup"))
    for i in range(c["n_folds"]):
        s = b0 + i * h
        e = bend if i == c["n_folds"] - 1 else s + h - pd.Timedelta(days=1)
        out.append(Fold(i, s, e, "tune" if i < c["tuning_folds"] else "honest"))
    return out


def peak_mask(df: pd.DataFrame) -> np.ndarray:
    return ((df.calendar_class == "workday") & df.hour.isin(config()["forecast"]["peak_hours"])).to_numpy()


def hour_group(df: pd.DataFrame) -> np.ndarray:
    pk = peak_mask(df)
    night = df.hour.isin([23, 0, 1, 2, 3, 4, 5, 6]).to_numpy()
    return np.where(pk, "peak", np.where(night, "night", "day"))


# ------------------------------------------------------------------------------------------ metrics
def wape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.abs(y - p).sum() / max(y.sum(), 1e-9))


def poisson_dev(y, p):
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1e-6, None)
    t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / p), 0.0)
    return float(np.mean(2 * (t - (y - p))))


def pinball(y, q, tau):
    d = np.asarray(y, float) - np.asarray(q, float)
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def score(df: pd.DataFrame, col: str, ycol: str) -> dict:
    y, p = df[ycol].to_numpy(), df[col].to_numpy()
    pk = peak_mask(df)
    out = {"wape": wape(y, p), "peak_wape": wape(y[pk], p[pk]), "offpeak_wape": wape(y[~pk], p[~pk]),
           "poisson_dev": poisson_dev(y, p), "rmse": float(np.sqrt(np.mean((y - p) ** 2)))}
    if f"{col}_lo80" in df and df[f"{col}_lo80"].notna().all():
        lo80, hi80 = df[f"{col}_lo80"].to_numpy(), df[f"{col}_hi80"].to_numpy()
        lo90, hi90 = df[f"{col}_lo90"].to_numpy(), df[f"{col}_hi90"].to_numpy()
        out.update({"pinball_10": pinball(y, lo80, 0.1), "pinball_50": pinball(y, p, 0.5), "pinball_90": pinball(y, hi80, 0.9),
                    "cov80": float(np.mean((y >= lo80) & (y <= hi80))), "cov90": float(np.mean((y >= lo90) & (y <= hi90))),
                    "cov80_peak": float(np.mean((y[pk] >= lo80[pk]) & (y[pk] <= hi80[pk]))),
                    "cov90_peak": float(np.mean((y[pk] >= lo90[pk]) & (y[pk] <= hi90[pk]))),
                    "cov80_offpeak": float(np.mean((y[~pk] >= lo80[~pk]) & (y[~pk] <= hi80[~pk]))),
                    "cov90_offpeak": float(np.mean((y[~pk] >= lo90[~pk]) & (y[~pk] <= hi90[~pk])))})
    return out


# ------------------------------------------------------------------------------------------ features per fold
def prepare_fold(sig: pd.DataFrame, k: int, train_end) -> tuple[pd.DataFrame, arch.ArchetypeModel, pd.DataFrame, dict]:
    """Fit archetypes on training days only, attach archetype_prior, freeze growth. No future information."""
    train = sig[sig.ts < pd.Timestamp(train_end)]
    share, comp = arch.daily_shapes(train)
    model, _ = arch.fit(share, comp, k=k)
    labels = model.assign(share)
    di = arch.day_info(sig)
    prior = arch.build_prior(di, labels, train_end)
    out = S.freeze_growth(sig, train_end)
    out["archetype_prior"] = arch.apply_prior(out, prior).to_numpy()
    seg_cent = arch.segment_centroids(train, labels)
    return out, model, seg_cent, prior


def add_lags(sig: pd.DataFrame) -> pd.DataFrame:
    s = sig.set_index("ts")
    out = sig.copy()
    for seg in SEGMENTS:
        for lag in (24, 168):
            out[f"{seg}_lag{lag}"] = s[seg].reindex(sig.ts - pd.Timedelta(hours=lag)).to_numpy()
    daily = sig.groupby("date")[SEGMENTS].sum()
    for seg in SEGMENTS:
        out[f"{seg}_prevday"] = daily[seg].reindex(sig.date - pd.Timedelta(days=1)).to_numpy()
    return out


LAG_FEATURES = [f"{s}_{x}" for s in SEGMENTS for x in ("lag24", "lag168", "prevday")]


# ------------------------------------------------------------------------------------------ contenders
def seasonal_naive(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Same hour, same calendar class, previous comparable week (latest comparable day before the origin)."""
    tr = train[["date", "dow", "calendar_class", "hour"] + SEGMENTS]
    out = pd.DataFrame(index=test.index, columns=SEGMENTS, dtype=float)
    days = tr[["date", "dow", "calendar_class"]].drop_duplicates()
    for (dow, cls), idx in test.groupby(["dow", "calendar_class"]).groups.items():
        if cls in ("workday", "weekend"):
            cand = days[(days.dow == dow) & (days.calendar_class == cls)]
        else:
            cand = days[days.calendar_class == cls]
        if cand.empty:
            cand = days[days.dow == dow]
        ref = cand.date.max()
        prof = tr[tr.date == ref].set_index("hour")[SEGMENTS].reindex(range(24)).interpolate(limit_direction="both")
        out.loc[idx, SEGMENTS] = prof.loc[test.loc[idx, "hour"]].to_numpy()
    return out.astype(float)


def profile_level(train: pd.DataFrame, test: pd.DataFrame, seg_cent: pd.DataFrame) -> pd.DataFrame:
    """Archetype hourly shape x trailing 4-week daily mean (same working / non-working group)."""
    origin = train.date.max()
    recent = train[train.date > origin - pd.Timedelta(days=28)]
    daily = recent.groupby(["date", "workingday"])[SEGMENTS].sum().reset_index()
    level = daily.groupby("workingday")[SEGMENTS].mean()
    out = pd.DataFrame(index=test.index, columns=SEGMENTS, dtype=float)
    for seg in SEGMENTS:
        cent = seg_cent[seg_cent.segment == seg].set_index("archetype")[list(range(24))]
        sh = np.array([cent.loc[a, h] if a in cent.index else np.nan for a, h in zip(test.archetype_prior, test.hour, strict=True)])
        lv = level[seg].reindex(test.workingday).to_numpy()
        out[seg] = sh * lv
    return out.fillna(0.0)


NB_FORMULA = ("y ~ C(hour)*C(calendar_class) + C(weather_cat) + temp_c + I(temp_c**2) + hum_pct + wind_kmh"
              " + C(month) + growth_index")


def nb_glm(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    from .inference import nb_alpha

    out = pd.DataFrame(index=test.index, columns=SEGMENTS, dtype=float)
    seen = set(train.calendar_class)
    te = test.copy()
    te["calendar_class"] = te.calendar_class.where(te.calendar_class.isin(seen), "weekend")
    for seg in SEGMENTS:
        d = train.assign(y=train[seg])
        pois = smf.glm(NB_FORMULA, d, family=sm.families.Poisson()).fit()
        a = nb_alpha(d.y.to_numpy(), pois.fittedvalues.to_numpy())
        res = smf.glm(NB_FORMULA, d, family=sm.families.NegativeBinomial(alpha=a)).fit()
        out[seg] = res.predict(te).to_numpy()
    return out


def has_catboost() -> bool:
    try:
        import catboost  # noqa: F401

        return True
    except ImportError:
        return False


class HGBPoisson:
    """Fallback when CatBoost is not installed: scikit-learn HistGradientBoosting with Poisson loss and native
    categoricals (ordinal-encoded). Same fit/predict surface as the CatBoost path."""

    def __init__(self, depth, iterations, lr):
        from sklearn.ensemble import HistGradientBoostingRegressor

        self.m = HistGradientBoostingRegressor(loss="poisson", max_depth=depth, max_iter=iterations, learning_rate=lr,
                                               random_state=seed(), early_stopping=False)

    def _enc(self, X, fit=False):
        from sklearn.preprocessing import OrdinalEncoder

        cats = [c for c in X.columns if c in S.CATEGORICAL]
        X = X.copy()
        if fit:
            self.cats = cats
            self.enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan).fit(X[cats])
        X[self.cats] = self.enc.transform(X[self.cats])
        return X

    def fit(self, X, y):
        Xe = self._enc(X, fit=True)
        self.m.set_params(categorical_features=[c in self.cats for c in Xe.columns])
        self.m.fit(Xe, y)
        self.feature_names_ = list(X.columns)
        return self

    def predict(self, X):
        return self.m.predict(self._enc(X))


def catboost_fit(X: pd.DataFrame, y, loss: str, depth: int, iterations: int | None = None):
    c = config()["forecast"]["catboost"]
    if not has_catboost():
        return HGBPoisson(depth, iterations or c["iterations"] // 3, c["learning_rate"] * 2).fit(X, y)
    from catboost import CatBoostRegressor

    cats = [i for i, col in enumerate(X.columns) if col in S.CATEGORICAL]
    m = CatBoostRegressor(loss_function=loss, depth=depth, iterations=iterations or c["iterations"],
                          learning_rate=c["learning_rate"], random_seed=seed(), verbose=False, thread_count=-1,
                          allow_writing_files=False)
    m.fit(X, y, cat_features=cats)
    return m


def catboost_predict(m, X) -> np.ndarray:
    if isinstance(m, HGBPoisson):
        return np.asarray(m.predict(X), float)
    return np.asarray(m.predict(X, prediction_type="Exponent"), float)


def loss_grid() -> list[str]:
    c = config()["forecast"]["catboost"]
    if not has_catboost():
        return ["Poisson"]
    grid = [f"Tweedie:variance_power={p}" for p in c["variance_power_grid"]]
    if c.get("compare_poisson"):
        grid.append("Poisson")
    return grid


def model_backend() -> str:
    return "CatBoost" if has_catboost() else "scikit-learn HistGradientBoosting (Poisson) — CatBoost not installed"


# ------------------------------------------------------------------------------------------ conformal (MAPIE)
class Passthrough(BaseEstimator, RegressorMixin):
    """'Prefit' estimator whose prediction is column 0 of X (an out-of-sample forecast + 1).

    This lets MAPIE calibrate on walk-forward residuals from *earlier folds* without refitting."""

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        self.fitted_ = True
        return self

    def predict(self, X):
        return np.asarray(X, float)[:, 0]


def conformal_intervals(cal_pred, cal_y, test_pred, levels=(0.8, 0.9), cal_groups=None, test_groups=None) -> dict:
    """Gamma (relative) conformity score on (count + 1); Mondrian if groups given."""
    from mapie.conformity_scores import GammaConformityScore
    from mapie.regression import SplitConformalRegressor

    test_pred = np.asarray(test_pred, float)
    out = {lv: (np.full(len(test_pred), np.nan), np.full(len(test_pred), np.nan)) for lv in levels}
    if cal_groups is None:
        cal_groups = np.zeros(len(cal_pred))
        test_groups = np.zeros(len(test_pred))
    for g in np.unique(test_groups):
        cm, tm = cal_groups == g, test_groups == g
        if cm.sum() < 30:
            cm = np.ones(len(cal_pred), bool)
        est = Passthrough().fit(np.c_[np.asarray(cal_pred)[cm] + 1])
        mapie = SplitConformalRegressor(estimator=est, confidence_level=list(levels),
                                        conformity_score=GammaConformityScore(), prefit=True)
        mapie.conformalize(np.c_[np.asarray(cal_pred)[cm] + 1], np.asarray(cal_y, float)[cm] + 1)
        _, pis = mapie.predict_interval(np.c_[test_pred[tm] + 1])
        for j, lv in enumerate(levels):
            out[lv][0][tm] = np.clip(pis[:, 0, j] - 1, 0, None)
            out[lv][1][tm] = np.clip(pis[:, 1, j] - 1, 0, None)
    return out


# ------------------------------------------------------------------------------------------ backtest
def tune_catboost(sig: pd.DataFrame, folds: list[Fold], k: int) -> dict:
    """Hyper-parameters chosen on the first (tuning) folds only."""
    depths = config()["forecast"]["catboost"]["depth_grid"]
    rows = []
    for f in [f for f in folds if f.kind == "tune"]:
        df, *_ = prepare_fold(sig, k, f.start)
        tr = df[df.ts < f.start]
        te = df[(df.date >= f.start) & (df.date <= f.end) & (df.imputed_row == 0)]
        Xtr, Xte = S.model_matrix(tr), S.model_matrix(te)
        for seg in SEGMENTS:
            for loss in loss_grid():
                for d in depths:
                    m = catboost_fit(Xtr, tr[seg], loss, d)
                    p = catboost_predict(m, Xte)
                    pk = peak_mask(te)
                    rows.append({"fold": f.idx, "segment": seg, "loss": loss, "depth": d, "wape": wape(te[seg], p),
                                 "peak_wape": wape(te[seg].to_numpy()[pk], p[pk]), "poisson_dev": poisson_dev(te[seg], p)})
                    log.info("tune fold %d %s %s depth %d -> wape %.3f", f.idx, seg, loss, d, rows[-1]["wape"])
    t = pd.DataFrame(rows)
    save_csv(t, "catboost_tuning")
    agg = t.groupby(["segment", "loss", "depth"])[["wape", "peak_wape", "poisson_dev"]].mean().reset_index()
    best = {}
    for seg in SEGMENTS:
        a = agg[agg.segment == seg].sort_values(["poisson_dev"])
        best[seg] = {"loss": a.iloc[0].loss, "depth": int(a.iloc[0].depth)}
    return best


def daily_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("date").agg(dow=("dow", "first"), month=("month", "first"), calendar_class=("calendar_class", "first"),
                               archetype_prior=("archetype_prior", "first"), weather_cat=("weather_cat", "max"),
                               temp_c=("temp_c", "mean"), hum_pct=("hum_pct", "mean"), wind_kmh=("wind_kmh", "mean"),
                               wet_hours=("wet", "sum"), growth_index=("growth_index", "first"),
                               casual=("casual", "sum"), registered=("registered", "sum"), cnt=("cnt", "sum")).reset_index()
    return g


DAILY_FEATURES = ["dow", "month", "calendar_class", "archetype_prior", "weather_cat", "temp_c", "hum_pct", "wind_kmh",
                  "wet_hours", "growth_index"]


def _daily_X(d):
    X = d[DAILY_FEATURES].copy()
    for c in ["dow", "month", "calendar_class", "archetype_prior", "weather_cat"]:
        X[c] = X[c].astype(str)
    return X


def run_backtest(sig: pd.DataFrame, k: int) -> dict:
    folds = make_folds()
    best = tune_catboost(sig, folds, k)
    log.info("CatBoost tuned params: %s", best)
    lagged = add_lags(sig)
    preds, daily_rows = [], []
    for f in folds:
        log.info("Fold %d (%s) %s -> %s", f.idx, f.kind, f.start.date(), f.end.date())
        df, model, seg_cent, prior = prepare_fold(sig, k, f.start)
        df[LAG_FEATURES] = lagged[LAG_FEATURES].to_numpy()
        tr = df[df.ts < f.start]
        te = df[(df.date >= f.start) & (df.date <= f.end)].copy()
        P = {"seasonal_naive": seasonal_naive(tr, te), "profile_level": profile_level(tr, te, seg_cent), "nb_glm": nb_glm(tr, te)}
        Xtr, Xte = S.model_matrix(tr), S.model_matrix(te)
        cb = pd.DataFrame(index=te.index, columns=SEGMENTS, dtype=float)
        cbl = cb.copy()
        trl = tr.dropna(subset=LAG_FEATURES)
        for seg in SEGMENTS:
            m = catboost_fit(Xtr, tr[seg], best[seg]["loss"], best[seg]["depth"])
            cb[seg] = catboost_predict(m, Xte)
            ml = catboost_fit(S.model_matrix(trl, S.FEATURES + LAG_FEATURES), trl[seg], best[seg]["loss"], best[seg]["depth"])
            cbl[seg] = catboost_predict(ml, S.model_matrix(te, S.FEATURES + LAG_FEATURES))
        P["catboost_tweedie"], P["catboost_lags"] = cb, cbl
        # two-level consistency: daily model on day-level data
        dtr, dte = daily_model_frame(tr), daily_model_frame(te)
        dm = catboost_fit(_daily_X(dtr), dtr.cnt, best["registered"]["loss"], 6, iterations=600)
        dte["daily_pred"] = catboost_predict(dm, _daily_X(dte))
        for name, p in P.items():
            fr = te[["ts", "date", "dteday", "hour", "calendar_class", "workingday", "imputed_row", "disruption_day"] + SEGMENTS].copy()
            fr["fold"], fr["fold_kind"], fr["contender"] = f.idx, f.kind, name
            for seg in SEGMENTS:
                fr[f"pred_{seg}"] = np.clip(p[seg].to_numpy(), 0, None)
            fr["cnt"] = fr.casual + fr.registered
            fr["pred_cnt"] = fr.pred_casual + fr.pred_registered
            preds.append(fr)
        hsum = P["catboost_tweedie"].sum(axis=1).groupby(te.date).sum()
        dte = dte.set_index("date")
        dte["hourly_sum"] = hsum.reindex(dte.index)
        dte["fold"], dte["fold_kind"] = f.idx, f.kind
        daily_rows.append(dte[["fold", "fold_kind", "cnt", "daily_pred", "hourly_sum"]].reset_index())
    bp = pd.concat(preds, ignore_index=True)
    daily = pd.concat(daily_rows, ignore_index=True)
    return {"preds": bp, "daily": daily, "best": best, "folds": folds}


def add_intervals(bp: pd.DataFrame, mondrian: bool) -> pd.DataFrame:
    """For each contender/fold: calibrate on all earlier folds' out-of-sample rows (observed hours only)."""
    out = []
    levels = config()["forecast"]["conformal"]["levels"]
    for _name, g in bp.groupby("contender"):
        g = g.copy()
        g["hgroup"] = hour_group(g)
        for tgt in ["casual", "registered", "cnt"]:
            for lv in levels:
                tag = int(lv * 100)
                g[f"pred_{tgt}_lo{tag}"] = np.nan
                g[f"pred_{tgt}_hi{tag}"] = np.nan
        for fi in sorted(g.fold.unique()):
            cal = g[(g.fold < fi) & (g.imputed_row == 0)]
            if len(cal) < 500:
                continue
            tm = g.fold == fi
            for tgt in ["casual", "registered", "cnt"]:
                iv = conformal_intervals(cal[f"pred_{tgt}"], cal[tgt], g.loc[tm, f"pred_{tgt}"], tuple(levels),
                                         cal.hgroup.to_numpy() if mondrian else None,
                                         g.loc[tm, "hgroup"].to_numpy() if mondrian else None)
                for lv in levels:
                    tag = int(lv * 100)
                    g.loc[tm, f"pred_{tgt}_lo{tag}"] = iv[lv][0]
                    g.loc[tm, f"pred_{tgt}_hi{tag}"] = iv[lv][1]
        out.append(g)
    return pd.concat(out, ignore_index=True)


def score_all(bp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ev = bp[(bp.fold >= 0) & (bp.imputed_row == 0)]
    for (name, fi), g in ev.groupby(["contender", "fold"]):
        for tgt in ["cnt", "casual", "registered"]:
            r = score(g, f"pred_{tgt}", tgt)
            rows.append({"contender": name, "fold": fi, "fold_kind": g.fold_kind.iloc[0], "target": tgt, **r})
    sc = pd.DataFrame(rows)
    naive = sc[sc.contender == "seasonal_naive"].set_index(["fold", "target"])[["wape", "peak_wape"]]
    sc = sc.join(naive, on=["fold", "target"], rsuffix="_naive")
    sc["skill_vs_naive"] = 1 - sc.wape / sc.wape_naive
    sc["peak_skill_vs_naive"] = 1 - sc.peak_wape / sc.peak_wape_naive
    return sc.drop(columns=["wape_naive", "peak_wape_naive"])


def pooled_scores(bp: pd.DataFrame, kinds=("honest",)) -> pd.DataFrame:
    ev = bp[bp.fold_kind.isin(kinds) & (bp.imputed_row == 0)]
    rows = []
    for name, g in ev.groupby("contender"):
        for tgt in ["cnt", "casual", "registered"]:
            rows.append({"contender": name, "target": tgt, **score(g, f"pred_{tgt}", tgt)})
    p = pd.DataFrame(rows)
    nv = p[p.contender == "seasonal_naive"].set_index("target")
    p["skill_vs_naive"] = 1 - p.wape / p.target.map(nv.wape)
    p["peak_skill_vs_naive"] = 1 - p.peak_wape / p.target.map(nv.peak_wape)
    return p


# ------------------------------------------------------------------------------------------ figures
def fig_backtest(sc: pd.DataFrame):
    d = sc[sc.target == "cnt"]
    return alt.Chart(d).mark_line(point=True).encode(
        x=alt.X("fold:O"), y=alt.Y("peak_wape:Q", title="peak-window WAPE (total rentals)", axis=alt.Axis(format="%")),
        color=alt.Color("contender:N"), strokeDash=alt.condition(alt.datum.contender == "catboost_lags", alt.value([4, 3]), alt.value([1, 0])),
    ).properties(width=560, height=240, title="Walk-forward backtest: peak-window WAPE by fold (folds 0-1 = tuning)")


def fig_week(bp: pd.DataFrame, champion: str):
    d = bp[(bp.contender == champion) & (bp.fold >= 0)]
    wk = d[(d.ts >= "2012-06-04") & (d.ts < "2012-06-11")]
    if wk.empty:
        wk = d.head(168)
    band = alt.Chart(wk).mark_area(opacity=0.25, color="#1b6ca8").encode(x=alt.X("ts:T", title=None), y=alt.Y("pred_cnt_lo90:Q", title="rentals per hour"), y2="pred_cnt_hi90:Q")
    band80 = alt.Chart(wk).mark_area(opacity=0.35, color="#1b6ca8").encode(x="ts:T", y="pred_cnt_lo80:Q", y2="pred_cnt_hi80:Q")
    line = alt.Chart(wk).mark_line(color="#1b6ca8").encode(x="ts:T", y="pred_cnt:Q")
    obs = alt.Chart(wk).mark_point(color="#222", size=12, filled=True).encode(x="ts:T", y="cnt:Q")
    return (band + band80 + line + obs).properties(width=600, height=240, title=f"{champion}: forecast with 80% / 90% conformal bands vs actual (one June 2012 week)")


def fig_coverage(cov: pd.DataFrame):
    return alt.Chart(cov).mark_bar().encode(
        x=alt.X("group:N", title=None), y=alt.Y("coverage:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0.5, 1])),
        color="level:N", xOffset="level:N",
    ).properties(width=420, height=220, title="Empirical interval coverage on the backtest (champion, total rentals)")


# ------------------------------------------------------------------------------------------ champion + final
def choose_champion(pooled: pd.DataFrame) -> tuple[str, dict]:
    t = pooled[(pooled.target == "cnt") & ~pooled.contender.isin(DIAGNOSTIC)].set_index("contender")
    champ = t.peak_wape.idxmin()
    beats = {c: bool(t.loc[champ, "peak_wape"] < t.loc[c, "peak_wape"]) for c in SIMPLE if c != champ}
    return champ, beats


def coverage_table(bp: pd.DataFrame, champion: str) -> pd.DataFrame:
    ev = bp[(bp.contender == champion) & (bp.fold >= 0) & (bp.imputed_row == 0)]
    pk = peak_mask(ev)
    rows = []
    for tgt in ["cnt", "casual", "registered"]:
        y = ev[tgt].to_numpy()
        for lv in (80, 90):
            lo, hi = ev[f"pred_{tgt}_lo{lv}"].to_numpy(), ev[f"pred_{tgt}_hi{lv}"].to_numpy()
            inside = (y >= lo) & (y <= hi)
            for gname, m in [("all hours", np.ones(len(y), bool)), ("peak", pk), ("off-peak", ~pk)]:
                rows.append({"target": tgt, "level": f"{lv}%", "nominal": lv / 100, "group": gname,
                             "coverage": float(inside[m].mean()), "mean_width": float((hi - lo)[m].mean())})
    c = pd.DataFrame(rows)
    c["within_3pts"] = (c.coverage - c.nominal).abs() <= 0.03
    return c


def residual_quantiles(bp: pd.DataFrame, champion: str) -> pd.DataFrame:
    """Signed relative residuals r = (y+1)/(pred+1) - 1 per hour group: the predictive distribution used by
    the Monte Carlo (all walk-forward out-of-sample rows)."""
    ev = bp[(bp.contender == champion) & (bp.imputed_row == 0)].copy()
    ev["hgroup"] = hour_group(ev)
    qs = np.linspace(0.005, 0.995, 199)
    rows = []
    for tgt in ["cnt", "casual", "registered"]:
        r = (ev[tgt] + 1) / (ev[f"pred_{tgt}"] + 1) - 1
        for g in ["peak", "day", "night"]:
            v = r[ev.hgroup == g]
            rows += [{"target": tgt, "hgroup": g, "q": float(q), "r": float(np.quantile(v, q))} for q in qs]
    return pd.DataFrame(rows)


def intraday_corr(bp: pd.DataFrame, champion: str) -> float:
    from scipy.stats import norm

    ev = bp[(bp.contender == champion) & (bp.imputed_row == 0) & bp.hour.between(7, 22)].copy()
    ev["r"] = (ev.cnt + 1) / (ev.pred_cnt + 1)
    ev["hgroup"] = hour_group(ev)
    ev["u"] = ev.groupby("hgroup").r.rank(pct=True).clip(0.001, 0.999)
    ev["z"] = norm.ppf(ev.u)
    g = ev.groupby("dteday").z.agg(["sum", "size"])
    g = g[g["size"] == 16]
    n = 16
    rho = ((g["sum"] ** 2).mean() / n - 1) / (n - 1)
    return float(np.clip(rho, 0, 0.99)) if np.isfinite(rho) else config()["concurrency"]["intraday_corr_fallback"]


def _approx_contributions(m, Xs: pd.DataFrame) -> np.ndarray:
    """SHAP-like per-row contributions without the shap library: log-prediction change when a feature is replaced by
    a random draw from the sample (one-at-a-time interventional approximation)."""
    rng = np.random.default_rng(seed())
    base = np.log(np.clip(catboost_predict(m, Xs), 1e-6, None))
    out = np.zeros((len(Xs), Xs.shape[1]))
    for j, col in enumerate(Xs.columns):
        Xp = Xs.copy()
        Xp[col] = Xs[col].to_numpy()[rng.permutation(len(Xs))]
        out[:, j] = base - np.log(np.clip(catboost_predict(m, Xp), 1e-6, None))
    return out


def fit_final(sig: pd.DataFrame, k: int, best: dict, champion: str) -> dict:
    """Champion refit on all 2011-2012 data for the demand cube + SHAP driver analysis."""
    end = pd.Timestamp(config()["data"]["end"]) + pd.Timedelta(days=1)
    df, model, seg_cent, prior = prepare_fold(sig, k, end)
    X = S.model_matrix(df)
    models = {}
    shap_rows, dep_rows = [], []
    for seg in SEGMENTS:
        m = catboost_fit(X, df[seg], best[seg]["loss"], best[seg]["depth"])
        models[seg] = m
        samp = df[df.yr == 1].sample(min(4000, int((df.yr == 1).sum())), random_state=seed())
        Xs = S.model_matrix(samp)
        if has_catboost():
            from catboost import Pool

            m.save_model(str(path("models") / f"catboost_{seg}.cbm"))
            sv = m.get_feature_importance(Pool(Xs, cat_features=[c for c in Xs.columns if c in S.CATEGORICAL]), type="ShapValues")[:, :-1]
        else:
            import joblib

            joblib.dump(m, path("models") / f"hgb_{seg}.joblib")
            sv = _approx_contributions(m, Xs)
        for j, col in enumerate(Xs.columns):
            shap_rows.append({"segment": seg, "feature": col, "mean_abs_shap": float(np.abs(sv[:, j]).mean())})
        for col in ["temp_c", "hum_pct", "hour", "comfort_index"]:
            j = list(Xs.columns).index(col)
            dep_rows.append(pd.DataFrame({"segment": seg, "feature": col, "value": samp[col].astype(float).to_numpy(), "shap": sv[:, j]}))
    shap = pd.DataFrame(shap_rows)
    shap["rank"] = shap.groupby("segment").mean_abs_shap.rank(ascending=False).astype(int)
    save_csv(shap.sort_values(["segment", "rank"]), "shap_summary")
    dep = pd.concat(dep_rows, ignore_index=True)
    save_csv(dep, "shap_dependence")
    ch = alt.Chart(shap).mark_bar().encode(y=alt.Y("feature:N", sort="-x", title=None), x=alt.X("mean_abs_shap:Q", title="mean |SHAP| (log-rate units)"),
                                          color=alt.Color("segment:N", scale=alt.Scale(domain=["casual", "registered"], range=["#d95f02", "#1b6ca8"])), yOffset="segment:N")
    save_chart(ch.properties(width=420, height=320, title="Drivers of demand (CatBoost SHAP, 2012 sample)"), "ws3_shap_summary")
    td = dep[dep.feature == "temp_c"]
    ch2 = alt.Chart(td).mark_circle(size=8, opacity=0.35).encode(x=alt.X("value:Q", title="temperature °C"), y=alt.Y("shap:Q", title="SHAP (log-rate)"),
                                                               color=alt.Color("segment:N", scale=alt.Scale(domain=["casual", "registered"], range=["#d95f02", "#1b6ca8"])))
    save_chart(ch2.properties(width=480, height=240, title="Temperature effect (SHAP dependence)"), "ws3_shap_temperature")
    save_json({"k": k, "prior_train_end": prior["train_end"], "best": best, "champion": champion}, "final_model_meta")
    return {"models": models, "prior": prior, "seg_cent": seg_cent, "frame": df}


def run(sig: pd.DataFrame, k: int) -> dict:
    out = run_backtest(sig, k)
    bp = out["preds"]
    bp_g = add_intervals(bp, mondrian=False)
    pooled_g = pooled_scores(bp_g)
    champion, beats = choose_champion(pooled_g)
    cov_g = coverage_table(bp_g, champion)
    pk_rows = cov_g[(cov_g.target == "cnt") & (cov_g.group == "peak")]
    peak_under = bool(((pk_rows.coverage - pk_rows.nominal).abs() > 0.03).any())  # peaks mis-calibrated either way
    method = "global split-conformal"
    bp_final, cov = bp_g, cov_g
    if peak_under:
        log.info("Peak coverage off by > 3 points with global conformal -> Mondrian by hour group")
        bp_m = add_intervals(bp, mondrian=True)
        cov_m = coverage_table(bp_m, champion)
        bp_final, cov, method = bp_m, cov_m, "Mondrian split-conformal (hour groups: peak / day / night)"
    save_csv(cov_g, "coverage_global")
    save_csv(cov, "coverage")
    sc = score_all(bp_final)
    save_csv(sc, "backtest_scores")
    pooled = pooled_scores(bp_final)
    save_csv(pooled, "backtest_scores_pooled_honest")
    save_csv(pooled_scores(bp_final, kinds=("tune", "honest")), "backtest_scores_pooled_all")

    daily = out["daily"]
    daily = daily[daily.fold >= 0]
    cons = daily.groupby("fold").apply(lambda d: pd.Series({
        "daily_model_wape": wape(d.cnt, d.daily_pred), "hourly_sum_wape": wape(d.cnt, d.hourly_sum),
        "disagreement": float(np.abs(d.hourly_sum - d.daily_pred).sum() / d.daily_pred.sum())}), include_groups=False).reset_index()
    # optional reconciliation: scale hourly champion shapes to the daily total; keep only if it helps on tuning folds
    ch = bp_final[(bp_final.contender == "catboost_tweedie") & (bp_final.fold >= 0)].merge(
        daily[["date", "daily_pred"]], on="date", how="left")
    hs = ch.groupby("date").pred_cnt.transform("sum")
    ch["rec_cnt"] = ch.pred_cnt * (ch.daily_pred / hs.replace(0, np.nan)).fillna(1)
    rec = []
    for fi, g in ch[ch.imputed_row == 0].groupby("fold"):
        pk = peak_mask(g)
        rec.append({"fold": fi, "wape_base": wape(g.cnt, g.pred_cnt), "wape_reconciled": wape(g.cnt, g.rec_cnt),
                    "peak_wape_base": wape(g.cnt[pk], g.pred_cnt[pk]), "peak_wape_reconciled": wape(g.cnt[pk], g.rec_cnt[pk])})
    rec = pd.DataFrame(rec)
    cons = cons.merge(rec, on="fold")
    save_csv(cons, "two_level_consistency")
    tune_rec = cons[cons.fold < config()["forecast"]["tuning_folds"]]
    keep_rec = bool(tune_rec.wape_reconciled.mean() < tune_rec.wape_base.mean())

    con = connect()
    write_table(con, "backtest_predictions", bp_final.drop(columns=["hgroup"], errors="ignore"))
    con.close()

    rq = residual_quantiles(bp_final, champion)
    save_csv(rq, "residual_quantiles")
    rho = intraday_corr(bp_final, champion)
    save_chart(fig_backtest(sc), "ws3_backtest_peak_wape")
    save_chart(fig_week(bp_final, champion), "ws3_forecast_week")
    save_chart(fig_coverage(cov[cov.target == "cnt"]), "ws3_interval_coverage")

    final = fit_final(sig, k, out["best"], champion)
    sens = disruption_sensitivity(sig, k, out["best"])
    summ = {"model_backend": model_backend(), "champion": champion, "beats_simpler_on_peak_wape": beats, "catboost_params": out["best"],
            "interval_method": method, "peaks_under_covered_globally": peak_under, "intraday_corr": rho,
            "reconciliation_kept": keep_rec, "coverage_within_3pts_all": bool(cov[cov.target == "cnt"].within_3pts.all()),
            "pooled_honest": pooled[pooled.target == "cnt"].set_index("contender")[["wape", "peak_wape", "skill_vs_naive", "peak_skill_vs_naive"]].round(4).to_dict(orient="index"),
            "disruption_sensitivity": sens}
    save_json(summ, "ws3_forecast_summary")
    return {**summ, "final": final, "bp": bp_final, "residual_q": rq}


def disruption_sensitivity(sig: pd.DataFrame, k: int, best: dict) -> dict:
    """Champion with disruption days removed from training, scored on the honest folds (WS1 sensitivity test)."""
    rows = []
    for f in [f for f in make_folds() if f.kind == "honest"][::3]:
        df, *_ = prepare_fold(sig, k, f.start)
        te = df[(df.date >= f.start) & (df.date <= f.end) & (df.imputed_row == 0)]
        for excl in (False, True):
            tr = df[(df.ts < f.start) & ((df.disruption_day == 0) if excl else True)]
            p = sum(catboost_predict(catboost_fit(S.model_matrix(tr), tr[s], best[s]["loss"], best[s]["depth"]), S.model_matrix(te)) for s in SEGMENTS)
            rows.append({"fold": f.idx, "exclude_disruption_days": excl, "wape": wape(te.cnt, p)})
    d = pd.DataFrame(rows)
    save_csv(d, "disruption_sensitivity")
    g = d.groupby("exclude_disruption_days").wape.mean()
    return {"wape_with": float(g.loc[False]), "wape_without": float(g.loc[True])}
