"""Day archetypes: cluster each day's 24-hour *shape* (hourly share of the day's total)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .common import config, seed

HOURS = list(range(24))


def daily_shapes(sig: pd.DataFrame, value: str = "cnt") -> tuple[pd.DataFrame, pd.Series]:
    """Returns (share matrix date x 24 with NaN for missing hours, complete-day mask)."""
    piv = sig.pivot_table(index="dteday", columns="hour", values=value, aggfunc="sum").reindex(columns=HOURS)
    complete = piv.notna().all(axis=1)
    tot = piv.sum(axis=1, min_count=1)
    share = piv.div(tot.replace(0, np.nan), axis=0)
    return share, complete


@dataclass
class ArchetypeModel:
    km: KMeans
    k: int
    names: dict  # cluster id -> name
    centroids: pd.DataFrame  # name x 24 shares (cnt)

    def assign(self, share: pd.DataFrame) -> pd.Series:
        """Nearest centroid on observed hours only (renormalised), so partial days get a label too."""
        C = self.km.cluster_centers_
        out = []
        for _, row in share.iterrows():
            m = row.notna().to_numpy()
            if m.sum() < 12 or not np.isfinite(row[m]).all():
                out.append(None)
                continue
            x = row.to_numpy()[m]
            x = x / x.sum() if x.sum() > 0 else x
            Cm = C[:, m]
            Cm = Cm / Cm.sum(axis=1, keepdims=True)
            out.append(self.names[int(np.argmin(((Cm - x) ** 2).sum(axis=1)))])
        return pd.Series(out, index=share.index, name="archetype")


def _name_clusters(C: np.ndarray) -> dict:
    am = C[:, 7:10].sum(1)
    pm = C[:, 16:20].sum(1)
    mid = C[:, 10:16].sum(1)
    night = np.r_[C[:, 0:6].sum(1)] if C.ndim == 2 else 0
    names = {}
    commute = int(np.argmax(am + pm - mid))
    leisure = int(np.argmax(mid - am))
    names[commute] = "commute"
    if leisure != commute:
        names[leisure] = "leisure"
    for i in range(len(C)):
        if i in names:
            continue
        if night[i] > np.median(night) and night[i] == night.max():
            names[i] = "late-night"
        elif am[i] < am[commute] and pm[i] > mid[i] * 0.8:
            names[i] = "flattened-commute"
        else:
            names[i] = "flattened-leisure"
    # ensure unique
    seen = {}
    for i, n in list(names.items()):
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            names[i] = f"{n}-{seen[n]}"
    return names


def fit(share: pd.DataFrame, complete: pd.Series, k: int | None = None) -> tuple[ArchetypeModel, pd.DataFrame]:
    cfg = config()["archetypes"]
    X = share[complete].to_numpy()
    scores = []
    ks = [k] if k else cfg["k_range"]
    best = None
    for kk in ks:
        km = KMeans(n_clusters=kk, n_init=cfg["n_init"], random_state=seed()).fit(X)
        s = silhouette_score(X, km.labels_) if kk > 1 else np.nan
        scores.append({"k": kk, "silhouette": s, "inertia": km.inertia_})
        if best is None or s > best[1]:
            best = (km, s, kk)
    km, _, kbest = best
    names = _name_clusters(km.cluster_centers_)
    cent = pd.DataFrame(km.cluster_centers_, columns=HOURS, index=[names[i] for i in range(kbest)])
    return ArchetypeModel(km, kbest, names, cent), pd.DataFrame(scores)


def segment_centroids(sig: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Mean hourly share per archetype for each segment (used by the profile x level contender)."""
    rows = []
    for seg in ["casual", "registered", "cnt"]:
        sh, comp = daily_shapes(sig, seg)
        sh = sh[comp].join(labels, how="inner").dropna(subset=["archetype"])
        m = sh.groupby("archetype")[HOURS].mean()
        m = m.div(m.sum(axis=1), axis=0)
        m["segment"] = seg
        rows.append(m.reset_index())
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------------- prior
def build_prior(day_info: pd.DataFrame, labels: pd.Series, train_end) -> dict:
    """Most likely archetype for (dow, month, holiday) from TRAINING days only (date < train_end).

    day_info: dteday, date, dow, month, holiday, calendar_class. Fallbacks: (dow, holiday) then calendar_class."""
    d = day_info.merge(labels.rename("archetype"), left_on="dteday", right_index=True, how="inner")
    d = d[(d.date < pd.Timestamp(train_end)) & d.archetype.notna()]
    mode = lambda s: s.value_counts().idxmax()  # noqa: E731
    return {
        "full": d.groupby(["dow", "month", "holiday"]).archetype.agg(mode).to_dict(),
        "dow": d.groupby(["dow", "holiday"]).archetype.agg(mode).to_dict(),
        "class": d.groupby("calendar_class").archetype.agg(mode).to_dict(),
        "global": mode(d.archetype),
        "train_end": str(train_end),
        "n_days": int(len(d)),
    }


def apply_prior(df: pd.DataFrame, prior: dict) -> pd.Series:
    out = []
    for dow, month, hol, cls in zip(df.dow, df.month, df.holiday, df.calendar_class, strict=True):
        a = prior["full"].get((dow, month, hol))
        if a is None:
            a = prior["dow"].get((dow, hol))
        if a is None:
            a = prior["class"].get(cls, prior["global"])
        out.append(a)
    return pd.Series(out, index=df.index, name="archetype_prior")


def day_info(sig: pd.DataFrame) -> pd.DataFrame:
    return sig.groupby("dteday").agg(date=("date", "first"), dow=("dow", "first"), month=("month", "first"),
                                     holiday=("holiday", "first"), calendar_class=("calendar_class", "first"),
                                     workingday=("workingday", "first")).reset_index()
