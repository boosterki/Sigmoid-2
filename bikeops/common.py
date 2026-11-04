"""Shared helpers: configuration, paths, warehouse connection, artifact writers."""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("bikeops")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")


@lru_cache(maxsize=1)
def config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    p = ROOT / config()["paths"][key]
    if key != "warehouse":
        p.mkdir(parents=True, exist_ok=True)
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


def seed() -> int:
    return int(config()["project"]["seed"])


def connect(read_only: bool = False):
    import duckdb

    return duckdb.connect(str(path("warehouse")), read_only=read_only)


def write_table(con, name: str, df: pd.DataFrame) -> None:
    """Replace a warehouse table with a pandas frame."""
    con.register("_tmp_df", df)
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _tmp_df")
    con.unregister("_tmp_df")


def read_table(name: str, con=None) -> pd.DataFrame:
    own = con is None
    con = con or connect(read_only=True)
    try:
        return con.execute(f"SELECT * FROM {name}").df()
    finally:
        if own:
            con.close()


def save_csv(df: pd.DataFrame, name: str, index: bool = False) -> Path:
    p = path("tables") / f"{name}.csv"
    df.to_csv(p, index=index)
    return p


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(path("tables") / f"{name}.csv")


def save_json(obj, name: str) -> Path:
    p = path("tables") / f"{name}.json"
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    return p


def load_json(name: str):
    with open(path("tables") / f"{name}.json") as f:
        return json.load(f)


def _json_default(o):
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return str(o)


def save_chart(chart, name: str) -> Path:
    """Save an Altair chart as PNG (vl-convert) plus a self-contained HTML twin."""
    import altair as alt

    alt.data_transformers.disable_max_rows()
    p = path("figures") / f"{name}.png"
    chart.save(str(path("figures") / f"{name}.html"))
    try:
        import vl_convert  # noqa: F401

        chart.save(str(p), scale_factor=2)
    except ImportError:  # offline fallback: render with vendored vega JS in headless Chromium
        _png_via_browser(chart.to_dict(), p)
    return p


def _png_via_browser(spec: dict, out: Path) -> None:
    import base64

    from playwright.sync_api import sync_playwright

    vd = Path(__file__).parent / "_vendor"
    js = "".join((vd / f).read_text() for f in ["vega-5.25.0.js", "vega-lite-5.8.0.js", "vega-embed-6.22.1.js"])
    spec = dict(spec)
    spec.pop("$schema", None)
    html = f"<html><body><div id='v'></div><script>{js}</script></body></html>"
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page()
        pg.set_content(html)
        url = pg.evaluate("""async (spec) => { const r = await vegaEmbed('#v', spec, {actions:false, renderer:'canvas'});
                             return await r.view.toImageURL('png', 2); }""", spec)
        b.close()
    out.write_bytes(base64.b64decode(url.split(",", 1)[1]))


def md_table(df: pd.DataFrame, floatfmt: str = ".3g", index: bool = False) -> str:
    return df.to_markdown(index=index, floatfmt=floatfmt)


def env(key: str, default: str | None = None) -> str | None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except Exception:  # pragma: no cover
        pass
    return os.environ.get(key, default)
