"""Turn words into cube coordinates: dates -> (month, calendar_class), weather words -> weather band, times -> hour."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date

from ..signals import calendar_class_for_date

MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], start=1)}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})
MONTHS["sept"] = 9
WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
WEATHER = [
    (r"\b(rain\w*|wet|snow\w*|storm\w*|shower\w*|drizzl\w*|thunder\w*)\b", "wet"),
    (r"\b(hot|heat\w*|scorch\w*|sweltering|warmest)\b", "dry-hot"),
    (r"\b(cold|chilly|freez\w*|frost\w*|icy)\b", "dry-cold"),
    (r"\b(nice|mild|pleasant|sunny|dry|clear|fair|good weather)\b", "dry-mild"),
]


@dataclass
class Slots:
    month: int | None = None
    calendar_class: str | None = None
    weather_band: str | None = None
    hour: int | None = None
    date: str | None = None
    year: int | None = None
    service_level: float | None = None
    trip_median: float | None = None
    friction: float | None = None
    notes: list = field(default_factory=list)

    def dict(self):
        return {k: v for k, v in asdict(self).items() if v not in (None, [])}


def _parse_date(q: str) -> date | None:
    ql = q.lower()
    m = re.search(r"\b(20\d\d)-(\d{1,2})-(\d{1,2})\b", ql)
    if m:
        return date(int(m[1]), int(m[2]), int(m[3]))
    mon = "|".join(sorted(MONTHS, key=len, reverse=True))
    m = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({mon})\.?,?\s+(20\d\d)\b", ql)
    if m:
        return date(int(m[3]), MONTHS[m[2]], int(m[1]))
    m = re.search(rf"\b({mon})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(20\d\d)\b", ql)
    if m:
        return date(int(m[3]), MONTHS[m[1]], int(m[2]))
    return None


def _parse_hour(q: str) -> int | None:
    ql = q.lower()
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)", ql)
    if m:
        h = int(m[1]) % 12
        return h + 12 if m[3].startswith("p") else h
    m = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", ql)
    if m:
        return int(m[1])
    if "noon" in ql or "midday" in ql:
        return 12
    if "midnight" in ql:
        return 0
    return None


def resolve(q: str) -> Slots:
    s = Slots()
    ql = q.lower()
    d = _parse_date(q)
    if d:
        s.date, s.year, s.month = d.isoformat(), d.year, d.month
        s.calendar_class = calendar_class_for_date(d)
        if d.year > 2012:
            s.notes.append("Dates after 2012 are answered at the 2012 demand level.")
    else:
        for name, i in sorted(MONTHS.items(), key=lambda kv: -len(kv[0])):
            if re.search(rf"\b{name}\b", ql) and not (name == "may" and re.search(r"\bmay (i|we|you)\b", ql)):
                s.month = i
                break
        y = re.search(r"\b(2011|2012|20[1-9]\d)\b", ql)
        if y:
            s.year = int(y[1])
    if s.calendar_class is None:
        if re.search(r"\b(holiday|thanksgiving|christmas|independence day|labor day|memorial day)\b", ql):
            s.calendar_class = "holiday"
        elif re.search(r"\bbridge\b", ql):
            s.calendar_class = "bridge"
        elif re.search(r"\b(weekend|saturday|sunday|saturdays|sundays)\b", ql):
            s.calendar_class = "weekend"
        elif re.search(r"\b(weekday|workday|working day|work day|commute|monday|tuesday|wednesday|thursday|friday)s?\b", ql):
            s.calendar_class = "workday"
    for pat, band in WEATHER:
        if re.search(pat, ql):
            s.weather_band = band
            break
    s.hour = _parse_hour(q)
    m = re.search(r"\b(\d{2}(?:\.\d+)?)\s*%\s*(?:service|level|of days)?", ql)
    if m and ("service" in ql or "level" in ql):
        s.service_level = float(m[1]) / 100
    m = re.search(r"\b(\d{1,3})\s*(?:-| )?min(?:ute)?s?\b", ql)
    if m:
        s.trip_median = float(m[1])
    m = re.search(r"friction\s*(?:of\s*)?(\d(?:\.\d+)?)", ql)
    if m:
        s.friction = float(m[1])
    return s
