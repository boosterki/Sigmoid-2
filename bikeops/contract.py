"""Data contract (Readme encoded as pandera schemas).

`raw_schema()` describes what the Readme promises. Running it on raw hour.csv is *expected*
to fail on the sentinel zeros (hum = 0, windspeed = 0); those failures are evidence for the
audit and are written to artifacts/tables/contract_violations.csv.
`repaired_schema()` is the contract the cleaned table must satisfy.
"""
from __future__ import annotations

import pandas as pd

try:  # pandera >= 0.24 namespaces the pandas backend
    import pandera.pandas as pa
    from pandera.pandas import Check, Column, DataFrameSchema
except ImportError:  # pragma: no cover
    import pandera as pa
    from pandera import Check, Column, DataFrameSchema


def _calendar_checks() -> list:
    def _dt(df):
        return pd.to_datetime(df["dteday"])

    return [
        Check(lambda df: df["casual"] + df["registered"] == df["cnt"], name="casual_plus_registered_eq_cnt"),
        Check(lambda df: df["yr"] == _dt(df).dt.year - 2011, name="yr_matches_calendar"),
        Check(lambda df: df["mnth"] == _dt(df).dt.month, name="mnth_matches_calendar"),
        # Readme: weekday 0 = Sunday; pandas dayofweek 0 = Monday
        Check(lambda df: df["weekday"] == (_dt(df).dt.dayofweek + 1) % 7, name="weekday_matches_calendar"),
        Check(
            lambda df: df["workingday"] == ((df["weekday"].between(1, 5)) & (df["holiday"] == 0)).astype(int),
            name="workingday_matches_weekday_holiday",
        ),
    ]


def _columns(weathersit_allowed: list[int], positive_weather: bool) -> dict:
    frac_pos = [Check.gt(0), Check.le(1)] if positive_weather else [Check.ge(0), Check.le(1)]
    return {
        "instant": Column(int, Check.ge(1), required=False),
        "dteday": Column(str, Check.str_matches(r"^\d{4}-\d{2}-\d{2}$")),
        "season": Column(int, Check.isin([1, 2, 3, 4])),
        "yr": Column(int, Check.isin([0, 1])),
        "mnth": Column(int, Check.in_range(1, 12)),
        "hr": Column(int, Check.in_range(0, 23)),
        "holiday": Column(int, Check.isin([0, 1])),
        "weekday": Column(int, Check.in_range(0, 6)),
        "workingday": Column(int, Check.isin([0, 1])),
        "weathersit": Column(int, Check.isin(weathersit_allowed)),
        "temp": Column(float, [Check.gt(0), Check.le(1)]),
        "atemp": Column(float, [Check.ge(0), Check.le(1)], required=False),
        # hum = 0 % and windspeed = 0 km/h are physically implausible as hourly means in D.C.:
        "hum": Column(float, frac_pos),
        "windspeed": Column(float, frac_pos),
        "casual": Column(int, Check.ge(0)),
        "registered": Column(int, Check.ge(0)),
        "cnt": Column(int, Check.ge(0)),
    }


def raw_schema() -> DataFrameSchema:
    return DataFrameSchema(
        _columns([1, 2, 3, 4], positive_weather=True),
        checks=_calendar_checks(),
        unique=["dteday", "hr"],
        coerce=False,
        strict=False,
        name="hour_raw",
    )


def repaired_schema() -> DataFrameSchema:
    return DataFrameSchema(
        _columns([1, 2, 3], positive_weather=True),
        checks=_calendar_checks(),
        unique=["dteday", "hr"],
        coerce=False,
        strict=False,
        name="hour_repaired",
    )


def validate(df: pd.DataFrame, schema: DataFrameSchema) -> pd.DataFrame:
    """Validate lazily. Returns a tidy table of failure cases (empty if the frame passes)."""
    try:
        schema.validate(df, lazy=True)
        return pd.DataFrame(columns=["schema_context", "column", "check", "check_number", "failure_case", "index"])
    except pa.errors.SchemaErrors as err:
        fc = err.failure_cases.copy()
        keep = [c for c in ["schema_context", "column", "check", "check_number", "failure_case", "index"] if c in fc]
        return fc[keep].reset_index(drop=True)
