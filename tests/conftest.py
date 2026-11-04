import pandas as pd
import pytest

from bikeops.common import path


def _has_table(name: str) -> bool:
    if not path("warehouse").exists():
        return False
    from bikeops.common import connect

    con = connect(read_only=True)
    try:
        return name in set(con.execute("SELECT table_name FROM information_schema.tables").df().table_name)
    finally:
        con.close()


def require(*tables):
    missing = [t for t in tables if not _has_table(t)]
    if missing:
        pytest.skip(f"run `make all` first (missing warehouse tables: {missing})")


@pytest.fixture(scope="session")
def sig():
    require("signals")
    from bikeops.common import read_table

    s = read_table("signals")
    s["ts"] = pd.to_datetime(s.ts)
    s["date"] = pd.to_datetime(s.date)
    return s
