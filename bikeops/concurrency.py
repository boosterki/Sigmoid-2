"""Workstream 3.4 — from rentals per hour to bikes on the road (vectorised Monte Carlo).

For each simulated day:
    hourly rentals ~ forecast predictive distribution (conformal residual quantiles, Gaussian copula
                     with intraday correlation estimated from the backtest)
    start minute   ~ uniform within the hour
    trip duration  ~ LogNormal(median, p90)  (prior from config.yaml; the largest assumption)
    bikes_in_use(t)= number of trips active at minute t (trips crossing midnight wrap to the morning)
Common random numbers are used across trip medians, so results are monotone in the trip median.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .common import config

Z90 = norm.ppf(0.9)
MAX_TRIP_MIN = 240


def lognormal_params(median: float, p90: float | None = None) -> tuple[float, float]:
    c = config()["concurrency"]
    sigma = (np.log(c["trip_p90_min"]) - np.log(c["trip_median_min"])) / Z90 if p90 is None else (np.log(p90) - np.log(median)) / Z90
    return float(np.log(median)), float(sigma)


def draw_hourly_counts(pred: np.ndarray, hgroups: np.ndarray, rq: dict, rho: float, n: int, rng) -> np.ndarray:
    """pred: (24,) point forecast; rq: {hgroup: (q_levels, r_values)} signed relative residual quantiles."""
    zd = rng.standard_normal((n, 1))
    zh = rng.standard_normal((n, 24))
    u = norm.cdf(np.sqrt(rho) * zd + np.sqrt(1 - rho) * zh)
    r = np.empty((n, 24))
    for h in range(24):
        q, rv = rq[hgroups[h]]
        r[:, h] = np.interp(u[:, h], q, rv)
    counts = np.rint((pred[None, :] + 1) * (1 + r) - 1)
    return np.clip(counts, 0, None).astype(np.int64)


def bikes_in_use(counts: np.ndarray, trip_medians: list[float], rng, batch: int = 250) -> dict:
    """counts: (n, 24) simulated rentals. Returns per trip median: peak (n,), hourly_mean (n,24), hourly_max (n,24),
    minutes_above (callable data: minute-level not kept), utilisation needs fleet -> computed in fleet.py."""
    n = counts.shape[0]
    _, sigma = lognormal_params(18)
    out = {m: {"peak": np.empty(n), "hourly_mean": np.empty((n, 24), np.float32), "hourly_max": np.empty((n, 24), np.float32),
               "minute_hist": np.zeros(0)} for m in trip_medians}
    for b0 in range(0, n, batch):
        c = counts[b0:b0 + batch]
        nb = c.shape[0]
        flat = c.ravel()
        draw_id = np.repeat(np.repeat(np.arange(nb), 24), flat)
        hour_id = np.repeat(np.tile(np.arange(24), nb), flat)
        start = hour_id * 60 + rng.uniform(0, 60, size=len(hour_id))
        z = rng.standard_normal(len(hour_id))  # common random numbers across trip medians
        s_min = np.floor(start).astype(np.int64)
        for med in trip_medians:
            dur = np.minimum(np.exp(np.log(med) + sigma * z), MAX_TRIP_MIN)
            e_min = np.floor(start + dur).astype(np.int64)
            wrap = e_min >= 1440
            W = 1441
            idx_plus = draw_id * W + s_min
            idx_minus = draw_id * W + np.where(wrap, 1440, e_min)
            diff = np.bincount(idx_plus, minlength=nb * W) - np.bincount(idx_minus, minlength=nb * W)
            if wrap.any():
                diff += np.bincount(draw_id[wrap] * W, minlength=nb * W)
                diff -= np.bincount(draw_id[wrap] * W + (e_min[wrap] - 1440), minlength=nb * W)
            inuse = np.cumsum(diff.reshape(nb, W)[:, :1440], axis=1)
            o = out[med]
            o["peak"][b0:b0 + nb] = inuse.max(axis=1)
            hr = inuse.reshape(nb, 24, 60)
            o["hourly_mean"][b0:b0 + nb] = hr.mean(axis=2)
            o["hourly_max"][b0:b0 + nb] = hr.max(axis=2)
    return out


def simulate_cell(pred: np.ndarray, hgroups: np.ndarray, rq: dict, rho: float, trip_medians: list[float],
                  n: int | None = None, seed: int = 0) -> dict:
    n = n or config()["concurrency"]["n_draws"]
    rng = np.random.default_rng(seed)
    counts = draw_hourly_counts(pred, hgroups, rq, rho, n, rng)
    res = bikes_in_use(counts, trip_medians, rng)
    res["counts"] = counts
    return res


def simulate_trips_reference(counts_row: np.ndarray, median: float, rng) -> np.ndarray:
    """Slow, literal reference implementation (one day) used by tests to validate the vectorised kernel."""
    _, sigma = lognormal_params(18)
    inuse = np.zeros(1440, int)
    for h, c in enumerate(counts_row):
        for _ in range(int(c)):
            s = h * 60 + rng.uniform(0, 60)
            d = min(np.exp(np.log(median) + sigma * rng.standard_normal()), MAX_TRIP_MIN)
            a, b = int(np.floor(s)), int(np.floor(s + d))
            if b < 1440:
                inuse[a:b] += 1
            else:
                inuse[a:] += 1
                inuse[: b - 1440] += 1
    return inuse
