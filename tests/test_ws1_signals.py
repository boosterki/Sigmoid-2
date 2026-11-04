"""WS1 quality gate: contract behaviour, deterministic signals, no target leakage, training-only archetype prior."""
import numpy as np
import pandas as pd

from bikeops import archetypes as arch
from bikeops import contract
from bikeops import signals as S
from bikeops.common import ROOT, load_csv, read_table

from .conftest import require


def test_profile_facts_all_reproduced():
    require("raw_hour")
    pf = load_csv("profile_facts")
    assert pf.reproduced.all(), pf[~pf.reproduced]


def test_raw_contract_fails_only_on_sentinel_zeros():
    raw = pd.read_csv(ROOT / "raw" / "hour.csv")
    v = contract.validate(raw, contract.raw_schema())
    assert len(v) > 0
    assert set(v.column.dropna()) <= {"hum", "windspeed"}
    assert (v.column == "windspeed").sum() == 2180 and (v.column == "hum").sum() == 22


def test_repaired_passes_contract():
    require("hour_clean")
    clean = read_table("hour_clean")
    assert contract.validate(clean, contract.repaired_schema()).empty


def test_signals_deterministic():
    require("hour_clean")
    clean = read_table("hour_clean")
    a, b = S.build(clean), S.build(clean)
    pd.testing.assert_frame_equal(a, b)


def test_no_target_leakage(sig):
    assert not (set(S.FEATURES) & S.FORBIDDEN)
    clean = read_table("hour_clean")
    shuffled = clean.copy()
    rng = np.random.default_rng(0)
    for c in ["casual", "registered", "cnt"]:
        shuffled[c] = rng.permutation(shuffled[c].to_numpy())
    feats = [c for c in S.FEATURES if c != "archetype_prior"]
    pd.testing.assert_frame_equal(S.build(clean)[feats], S.build(shuffled)[feats])
    try:
        S.model_matrix(sig, feats + ["cnt"])
        raise AssertionError("model_matrix accepted a target column")
    except AssertionError as e:
        assert "forbidden" in str(e)


def test_archetype_prior_uses_training_days_only(sig):
    train_end = pd.Timestamp("2012-03-01")
    share, comp = arch.daily_shapes(sig[sig.ts < train_end])
    model, _ = arch.fit(share, comp, k=2)
    labels = model.assign(share)
    di = arch.day_info(sig)
    p1 = arch.build_prior(di, labels, train_end)
    # add (garbage) labels for future days: the prior must not change
    fut = pd.Series("leisure", index=di[di.date >= train_end].dteday, name="archetype")
    p2 = arch.build_prior(di, pd.concat([labels, fut]), train_end)
    assert p1["full"] == p2["full"] and p1["n_days"] == p2["n_days"]
    assert pd.Timestamp(p1["train_end"]) == train_end


def test_freeze_growth(sig):
    f = S.freeze_growth(sig, "2012-03-01")
    assert f.loc[f.ts >= "2012-03-01", "growth_index"].max() == sig.loc[sig.ts < "2012-03-01", "growth_index"].max()
