"""Feature construction — the single source of truth for every model input (Workstream 1.4).

Fold-dependent signals (archetype_prior, frozen growth_index) are added by functions that take
an explicit `train_end` so no information from the future can leak in.
"""
from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache

import numpy as np
import pandas as pd

from .common import config

CATEGORICAL = ["hour", "dow", "month", "met_season", "calendar_class", "archetype_prior", "weather_cat"]
NUMERIC = ["temp_c", "hum_pct", "wind_kmh", "comfort_index", "wet", "daylight", "growth_index"]
FEATURES = CATEGORICAL + NUMERIC
FORBIDDEN = {"instant", "cnt", "casual", "registered", "atemp", "atemp_c", "log_cnt"}
TARGETS = ["casual", "registered"]

MET_SEASON = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
              6: "summer", 7: "summer", 8: "summer", 9: "fall", 10: "fall", 11: "fall"}


# ------------------------------------------------------------------------------------- calendar
def calendar_class_for_days(days: pd.DataFrame) -> pd.Series:
    """days: one row per date, sorted, with columns date (datetime64), holiday (0/1).

    workday / weekend / holiday / bridge (a workday sandwiched between a holiday and a
    non-working day, e.g. the Friday after Thanksgiving)."""
    d = days.sort_values("date").reset_index(drop=True)
    wd = d.date.dt.dayofweek
    weekend = wd >= 5
    hol = d.holiday.astype(bool)
    nonwork = weekend | hol
    prev_nw = nonwork.shift(1, fill_value=False)
    next_nw = nonwork.shift(-1, fill_value=False)
    prev_h = hol.shift(1, fill_value=False)
    next_h = hol.shift(-1, fill_value=False)
    bridge = (~nonwork) & prev_nw & next_nw & (prev_h | next_h)
    cls = np.select([hol, weekend, bridge], ["holiday", "weekend", "bridge"], default="workday")
    return pd.Series(cls, index=d.date.values)


def us_federal_holidays(years) -> set:
    import holidays as hol

    out = set()
    for y in years:
        out |= set(hol.US(years=y).keys())
    return out


def calendar_class_for_date(d: date) -> str:
    """Resolver for arbitrary dates (desk): US federal holiday calendar."""
    d = pd.Timestamp(d)
    window = pd.date_range(d - pd.Timedelta(days=1), d + pd.Timedelta(days=1))
    hs = us_federal_holidays({x.year for x in window})
    days = pd.DataFrame({"date": window, "holiday": [int(x.date() in hs) for x in window]})
    return str(calendar_class_for_days(days).loc[d])


# ------------------------------------------------------------------------------------- daylight
@lru_cache(maxsize=1024)
def _sun_hours(d: date) -> tuple[float, float]:
    from zoneinfo import ZoneInfo

    from astral import LocationInfo
    from astral.sun import sun

    loc = config()["data"]["location"]
    li = LocationInfo(loc["name"], "USA", loc["timezone"], loc["latitude"], loc["longitude"])
    s = sun(li.observer, date=d, tzinfo=ZoneInfo(loc["timezone"]))
    return (s["sunrise"].hour + s["sunrise"].minute / 60, s["sunset"].hour + s["sunset"].minute / 60)


def daylight_flags(dates: pd.Series, hours: pd.Series) -> np.ndarray:
    uniq = {d: _sun_hours(d) for d in pd.to_datetime(dates).dt.date.unique()}
    rise = np.array([uniq[d][0] for d in pd.to_datetime(dates).dt.date])
    sett = np.array([uniq[d][1] for d in pd.to_datetime(dates).dt.date])
    mid = hours.to_numpy() + 0.5
    return ((mid >= rise) & (mid <= sett)).astype(int)


# ------------------------------------------------------------------------------------- comfort
def comfort_index(temp_c, hum_pct, wind_kmh):
    """comfort = temp_c - 0.15 * max(0, hum% - 70) - 0.25 * max(0, wind - 25 km/h).

    One 'is it pleasant to ride' number in deg C; coefficients in config.yaml."""
    c = config()["signals"]["comfort"]
    return (np.asarray(temp_c)
            - c["hum_penalty"] * np.maximum(0, np.asarray(hum_pct) - c["hum_threshold"])
            - c["wind_penalty"] * np.maximum(0, np.asarray(wind_kmh) - c["wind_threshold"]))


def weather_band(temp_c, wet) -> np.ndarray:
    b = config()["signals"]["weather_bands"]
    t = np.asarray(temp_c)
    return np.where(np.asarray(wet) == 1, "wet",
                    np.where(t >= b["hot_from_c"], "dry-hot", np.where(t < b["cold_below_c"], "dry-cold", "dry-mild")))


# ------------------------------------------------------------------------------------- build
def build(clean: pd.DataFrame) -> pd.DataFrame:
    """clean: hour_clean (repaired). Returns one row per observed (or night-lull) hour with signals."""
    sc = config()["data"]["scale"]
    days = (clean[["dteday", "holiday"]].drop_duplicates("dteday")
            .assign(date=lambda x: pd.to_datetime(x.dteday)).sort_values("date"))
    cls = calendar_class_for_days(days[["date", "holiday"]])
    cls_df = pd.DataFrame({"dteday": cls.index.strftime("%Y-%m-%d"), "calendar_class": cls.values})

    try:
        import polars as pl
    except ImportError:  # pandas fallback (identical columns)
        pl = None
    if pl is None:
        df = clean.copy()
        df["date"] = pd.to_datetime(df.dteday)
        df["hour"] = df.hr
        df["temp_c"] = df.temp * sc["temp"]
        df["hum_pct"] = df.hum * sc["hum"]
        df["wind_kmh"] = df.windspeed * sc["windspeed"]
        df["wet"] = (df.weathersit == 3).astype("int8")
        df["weather_cat"] = df.weathersit.astype("int64")
        df["growth_index"] = (df.yr * 12 + df.mnth - 1).astype(float)
        df["month"] = df.mnth
        df["dow"] = df.date.dt.dayofweek.astype("int64")
        df["met_season"] = df.month.map(MET_SEASON)
        df["ts"] = df.date + pd.to_timedelta(df.hour, unit="h")
        df = df.merge(cls_df, on="dteday", how="left")
        return _finish(df)
    lf = pl.from_pandas(clean).lazy()
    lf = (
        lf.with_columns(
            pl.col("dteday").str.strptime(pl.Date, "%Y-%m-%d").alias("date"),
            pl.col("hr").alias("hour"),
            (pl.col("temp") * sc["temp"]).alias("temp_c"),
            (pl.col("hum") * sc["hum"]).alias("hum_pct"),
            (pl.col("windspeed") * sc["windspeed"]).alias("wind_kmh"),
            (pl.col("weathersit") == 3).cast(pl.Int8).alias("wet"),
            pl.col("weathersit").cast(pl.Int64).alias("weather_cat"),
            (pl.col("yr") * 12 + pl.col("mnth") - 1).cast(pl.Float64).alias("growth_index"),
            pl.col("mnth").alias("month"),
        )
        .with_columns(
            (pl.col("date").dt.weekday() - 1).cast(pl.Int64).alias("dow"),  # 0 = Monday
            pl.col("month").replace_strict(MET_SEASON, return_dtype=pl.Utf8).alias("met_season"),
            (pl.col("date").cast(pl.Datetime) + pl.duration(hours=pl.col("hour"))).alias("ts"),
        )
        .join(pl.from_pandas(cls_df).lazy(), on="dteday", how="left")
    )
    df = lf.collect().to_pandas()
    return _finish(df)


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    df["comfort_index"] = comfort_index(df.temp_c, df.hum_pct, df.wind_kmh)
    df["daylight"] = daylight_flags(df["date"], df["hour"])
    df["weather_band"] = weather_band(df.temp_c, df.wet)
    keep = ["ts", "date", "dteday", "hour", "dow", "month", "yr", "met_season", "calendar_class", "holiday",
            "workingday", "weathersit", "weather_cat", "wet", "temp_c", "hum_pct", "wind_kmh", "comfort_index",
            "daylight", "growth_index", "weather_band", "imputed_row", "windspeed_imputed", "hum_imputed",
            "disruption_day", "casual", "registered", "cnt"]
    return df[keep].sort_values("ts").reset_index(drop=True)


def freeze_growth(df: pd.DataFrame, train_end) -> pd.DataFrame:
    """growth_index frozen at its last training value for later dates."""
    out = df.copy()
    cap = out.loc[out.ts < pd.Timestamp(train_end), "growth_index"].max()
    out["growth_index"] = out["growth_index"].clip(upper=cap)
    return out


def model_matrix(df: pd.DataFrame, features=None) -> pd.DataFrame:
    """Feature frame for CatBoost: categoricals as strings, numerics as float. Asserts no leakage."""
    features = features or FEATURES
    bad = FORBIDDEN & set(features)
    assert not bad, f"forbidden inputs requested: {bad}"
    X = df[features].copy()
    for c in features:
        if c in CATEGORICAL:
            X[c] = X[c].astype(str)
        else:
            X[c] = X[c].astype(float)
    return X


def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d")
