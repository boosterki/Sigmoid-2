"""Pipeline entry points (called by the Makefile): python -m bikeops.cli {ws1|ws2|ws3|ws4|memo}"""
from __future__ import annotations

import sys
import time

import pandas as pd

from .common import connect, load_json, log, read_table, save_json, write_table


def ws1():
    from . import archetypes as arch
    from . import audit, explore, reporting
    from . import signals as S

    summary = audit.run()
    clean = read_table("hour_clean")
    sig = S.build(clean)
    con = connect()
    write_table(con, "signals", sig)
    share, comp = arch.daily_shapes(sig)
    model, kscores = arch.fit(share, comp)
    labels = model.assign(share)
    write_table(con, "day_archetypes", labels.rename("archetype").reset_index().rename(columns={"index": "dteday"}))
    con.close()
    save_json({"k": model.k, "names": {str(k): v for k, v in model.names.items()}}, "ws1_archetypes")
    stats = explore.run(sig, model, labels, kscores)
    reporting.write_data_audit(summary, stats)
    log.info("WS1 done: k=%d archetypes, %d signal rows", model.k, len(sig))


def _sig() -> pd.DataFrame:
    sig = read_table("signals")
    sig["ts"] = pd.to_datetime(sig.ts)
    sig["date"] = pd.to_datetime(sig.date)
    return sig


def ws2():
    from . import inference, reporting

    s = inference.run(_sig())
    reporting.write_hypothesis_brief()
    log.info("WS2 done: %s", s["decision"])


def ws3():
    from . import cube, fleet, forecast, reporting

    sig = _sig()
    k = load_json("ws1_archetypes")["k"]
    res = forecast.run(sig, k)
    mondrian = res["interval_method"].startswith("Mondrian")
    scen = cube.scenario_frame(sig, res["final"]["prior"])
    preds = cube.predict_scenarios(scen, res["final"], res["bp"], res["champion"], mondrian)
    fm = cube.run_concurrency(sig, preds, res["residual_q"], res["intraday_corr"])
    fl = fleet.run(fm)
    reporting.write_model_card()
    log.info("WS3 done: champion=%s fleet=%s", res["champion"], fl["recommended"]["fleet"])


def ws4():
    from . import cube

    out = cube.materialise()
    save_json(out, "ws4_cube_summary")
    log.info("WS4 cube materialised: %s", out)


def memo():
    from . import cube, reporting

    reporting.write_decision_memo()
    con = connect()
    cube.build_notes(con)  # re-index notes so the desk can cite the memo too
    con.close()
    log.info("Decision memo written")


def main(argv=None):
    argv = argv or sys.argv[1:]
    cmds = {"ws1": ws1, "ws2": ws2, "ws3": ws3, "ws4": ws4, "memo": memo}
    for a in argv:
        t = time.time()
        cmds[a]()
        log.info("%s finished in %.0fs", a, time.time() - t)


if __name__ == "__main__":
    main()
