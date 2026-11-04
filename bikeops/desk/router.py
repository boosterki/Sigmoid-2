"""Intent routing: LLM with structured output, plus a deterministic rule router (offline + fallback)."""
from __future__ import annotations

import re

from . import llm

INTENTS = ["lookup", "compare", "what_if_fleet", "explain", "history"]

ROUTER_SYSTEM = """You route questions for a bike-share operations desk to exactly one intent:
- lookup: forecast/plan numbers for a type of day/hour/weather from the demand cube or the fleet utilisation table
  (e.g. bikes needed at 6 PM on a rainy weekday, busiest hours on a hot Saturday, idle share at 3 AM).
- compare: compare two groups or conditions (wet vs dry, casual vs registered, 2011 vs 2012, weekday vs weekend).
- what_if_fleet: fleet size under assumptions (service level, spatial friction / rebalancing, trip length, costs).
- explain: why/how questions about methods, assumptions, data cleaning, drivers of demand.
- history: what happened on a specific past date or period; records (highest / lowest days).
Return {"intent": <one of the five>, "reason": <short>}."""


def rule_route(q: str) -> str:
    ql = q.lower()
    if re.search(r"\b(why|how do you|how does the|what does .* mean|explain|assum\w*|method|treat|driver|drives|caveat)\b", ql):
        return "explain"
    if re.search(r"\b(what happened|happened on|on \d{1,2} \w+ 20\d\d|record|highest day|lowest day|busiest day|quietest day|in (?:the )?history|hurricane|outage)\b", ql) \
            or re.search(r"\b20(11|12)-\d{2}-\d{2}\b", ql):
        return "history"
    if re.search(r"\b(service level|rebalanc\w*|fleet (?:do|would|should) we need|how many bikes (?:do|would) we need|"
                 r"buy(?:ing)? (?:more )?bikes|friction|trips? (?:average|last|take)|what if|newsvendor|cost of a missed)\b", ql):
        return "what_if_fleet"
    if re.search(r"\b(compared?|versus|vs\.?|difference|smaller|larger|bigger|more than|less than|\w+er than|lower than|ratio|relative to)\b", ql):
        return "compare"
    return "lookup"


def route(q: str, provider: str | None = None) -> tuple[str, str]:
    """Returns (intent, how) where how in {'llm', 'rules'}."""
    if llm.provider_name(provider) != "rules" and llm.available(provider):
        try:
            out = llm.complete(ROUTER_SYSTEM, q, provider)
            it = str(out.get("intent", "")).strip()
            if it in INTENTS:
                return it, "llm"
        except Exception:
            pass
    return rule_route(q), "rules"
