"""WS3 quality gate: concurrency reproducible with seed; fleet monotone in service level and trip median; coverage."""
import numpy as np
import pytest

from bikeops import concurrency as cc
from bikeops.common import load_csv

from .conftest import require

PRED = np.array([10, 5, 3, 2, 3, 15, 80, 300, 550, 280, 150, 170, 210, 200, 190, 220, 380, 650, 600, 400, 280, 200, 140, 70], float)
HG = np.array(["night"] * 7 + ["peak"] * 3 + ["day"] * 6 + ["peak"] * 4 + ["day"] * 3 + ["night"])
Q = np.linspace(0.005, 0.995, 199)
RQ = {g: (Q, np.linspace(-0.4, 0.5, 199)) for g in ["peak", "day", "night"]}


def test_simulation_reproducible_with_seed():
    a = cc.simulate_cell(PRED, HG, RQ, 0.8, [18], n=200, seed=7)
    b = cc.simulate_cell(PRED, HG, RQ, 0.8, [18], n=200, seed=7)
    c = cc.simulate_cell(PRED, HG, RQ, 0.8, [18], n=200, seed=8)
    assert np.array_equal(a[18]["peak"], b[18]["peak"])
    assert not np.array_equal(a[18]["peak"], c[18]["peak"])


def test_simulation_monotone_in_trip_median():
    r = cc.simulate_cell(PRED, HG, RQ, 0.8, [12, 18, 25], n=300, seed=1)
    assert (r[12]["peak"] <= r[18]["peak"]).all() and (r[18]["peak"] <= r[25]["peak"]).all()


def test_vectorised_kernel_matches_reference():
    rng = np.random.default_rng(3)
    counts = np.rint(PRED).astype(int)[None, :]
    fast = cc.bikes_in_use(np.repeat(counts, 300, axis=0), [18], np.random.default_rng(4))[18]["hourly_mean"].mean(0)
    ref = np.mean([cc.simulate_trips_reference(counts[0], 18, rng).reshape(24, 60).mean(1) for _ in range(30)], axis=0)
    assert np.allclose(fast, ref, rtol=0.08, atol=3)


@pytest.fixture(scope="module")
def fm():
    require("concurrency_peaks", "day_mix")
    from bikeops.cube import load_fleet_model

    return load_fleet_model()


def test_fleet_monotone_in_alpha(fm):
    for t in (12, 18, 25):
        f = [fm.fleet(a, 1.3, t) for a in (0.90, 0.95, 0.99)]
        assert f == sorted(f)


def test_fleet_monotone_in_trip_median(fm):
    for a in (0.90, 0.95, 0.99):
        f = [fm.fleet(a, 1.3, t) for t in (12, 18, 25)]
        assert f == sorted(f)


def test_fleet_monotone_in_friction(fm):
    f = [fm.fleet(0.95, x, 18) for x in (1.2, 1.3, 1.4, 1.5)]
    assert f == sorted(f)


def test_interval_coverage_within_3_points():
    require("backtest_predictions")
    cov = load_csv("coverage")
    c = cov[(cov.target == "cnt") & (cov.group == "all hours")]
    assert ((c.coverage - c.nominal).abs() <= 0.03).all(), c


def test_backtest_scores_complete():
    require("backtest_predictions")
    sc = load_csv("backtest_scores")
    assert sc.fold.nunique() == 10
    assert set(sc.contender) == {"seasonal_naive", "profile_level", "nb_glm", "catboost_tweedie", "catboost_lags"}
