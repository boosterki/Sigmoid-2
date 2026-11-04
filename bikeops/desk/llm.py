"""Thin LiteLLM wrapper. Provider/model live in config.yaml; keys in .env. 'rules' = offline, no LLM."""
from __future__ import annotations

import json
import os
import re

from ..common import config, env


class LLMUnavailable(RuntimeError):
    pass


def provider_name(provider: str | None = None) -> str:
    return (provider or env("DESK_PROVIDER") or config()["desk"]["provider"]).lower()


def available(provider: str | None = None) -> bool:
    p = provider_name(provider)
    if p == "rules":
        return True
    if p == "anthropic":
        return bool(env("ANTHROPIC_API_KEY"))
    if p == "gemini":
        return bool(env("GEMINI_API_KEY"))
    if p == "ollama":
        try:
            import urllib.request

            urllib.request.urlopen(config()["desk"]["ollama_base_url"] + "/api/tags", timeout=1)
            return True
        except Exception:
            return False
    return False


def complete(system: str, user: str, provider: str | None = None, json_mode: bool = True) -> str | dict:
    p = provider_name(provider)
    if p == "rules":
        raise LLMUnavailable("rules provider has no LLM")
    try:
        import litellm
    except ImportError as e:  # pragma: no cover
        raise LLMUnavailable("litellm not installed") from e
    model = config()["desk"]["models"][p]
    kw = {}
    if p == "ollama":
        kw["api_base"] = config()["desk"]["ollama_base_url"]
    for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        v = env(k)
        if v:
            os.environ[k] = v
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if json_mode:
        msgs[0]["content"] += "\nReply with ONE JSON object only, no prose, no code fences."
    resp = litellm.completion(model=model, messages=msgs, temperature=config()["desk"]["temperature"], max_tokens=900, **kw)
    text = resp.choices[0].message.content or ""
    return parse_json(text) if json_mode else text.strip()


def parse_json(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))
