"""Chat loop: the model picks tools, Python runs them, the model phrases the exact results.

Default: Google Gemini (free tier) through its OpenAI-compatible endpoint.
Also works with the Anthropic API (Claude) or any other OpenAI-compatible API.
Configuration comes from Streamlit secrets or environment variables:

    LLM_PROVIDER = "gemini"               # or "anthropic" / "openai"
    LLM_API_KEY  = "..."                  # for Gemini: a key from https://aistudio.google.com/apikey
    LLM_MODEL    = "gemini-3.8-flash"     # optional for gemini/anthropic, required for "openai"
    LLM_BASE_URL = "https://..."          # only for other OpenAI-compatible providers
"""
import json
import os
import time

from analysis import STORES
from tools import run_tool, tool_specs

MAX_ROUNDS = 10
HISTORY_TURNS = 6
DEFAULT_MODELS = {"gemini": "gemini-3.8-flash", "anthropic": "claude-sonnet-5-5"}
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def load_config(secrets=None):
    def get(key, *alts):
        for k in (key, *alts):
            if secrets is not None:
                try:
                    if k in secrets:
                        return str(secrets[k])
                except Exception:  # noqa: BLE001 - no secrets file
                    pass
            if os.environ.get(k):
                return os.environ[k]
        return None

    provider = (get("LLM_PROVIDER") or "gemini").lower()
    own_key = {"gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(provider, "OPENAI_API_KEY")
    key = get("LLM_API_KEY", own_key)
    model = get("LLM_MODEL") or DEFAULT_MODELS.get(provider)
    base_url = get("LLM_BASE_URL") or (GEMINI_BASE_URL if provider == "gemini" else None)
    return {"provider": provider, "api_key": key, "model": model, "base_url": base_url}


def system_prompt(state):
    f = {t: v for t, v in state["filters"].items() if v}
    return f"""You are the analytics assistant of a bookstore dashboard with five stores: {', '.join(STORES)}.
Data: orders (one row per order), users (one row per user record; several records can be aliases of one real person),
books. Money is in USD; euro prices were converted at 1 EUR = 1.2 USD. Revenue = sum of paid_price.
"Unique/real users" = distinct person_id. "Best buyer" = the person with the highest total spending, reported with
ALL of their user_ids. "Most popular author" = the author set (one or several co-authors) with the most copies sold
(sum of quantity). Dates are YYYY-MM-DD.

The dashboard currently shows {state['store']}, view "{state['view']}", active filters: {json.dumps(f) if f else 'none'}.

RULES
1. Every number, id, name, date or list you report must come from a tool result in this conversation. Never guess,
   never estimate, never count or add in your head. For any arithmetic on tool results (differences, ratios,
   percentages, averages of results) call `calculate`.
2. If the user does not name a store, use {state['store']} and say which store the answer is for.
   For "all stores" / comparisons use store=ALL or call the tool for each store.
3. If a tool returns an error, fix the arguments and try again. If no tool can answer, say plainly that the data or
   tools do not support the question. Do not invent an answer.
4. If a result reports ties_for_first > 1, mention the tie.
5. Requests to change what the dashboard shows (filter, select, show, hide, sort, open, switch store, zoom the chart)
   are commands: use the control tools, then confirm in one or two sentences what is shown now, using rows_shown from
   the tool result. "Filter out X" / "select X" / "show X" means show only X, unless the user says hide, remove or
   exclude (then use negate=true); mention how to get the opposite.
6. When a filter needs world knowledge that is not in the data (e.g. "titles that contain a person's name"), call
   `list_values` for that column, decide which values match, then call `filter_table` with op "in" and the exact
   values in `values`. Say briefly that this selection is a judgment and give a few examples of what was included.
7. Answer concisely. Use short bullet lists for several items. Format money as $1,234.56."""


def _with_retry(call, attempts=3):
    """Free tiers allow only a few requests per minute: wait and retry on rate-limit errors."""
    for i in range(attempts):
        try:
            return call()
        except Exception as e:  # noqa: BLE001
            limited = "429" in str(e) or "rate" in str(e).lower() or "quota" in str(e).lower()
            if not limited or i == attempts - 1:
                raise
            time.sleep(10 * (i + 1))


def _anthropic_tools():
    return [{"name": t["name"], "description": t["description"], "input_schema": t["schema"]} for t in tool_specs()]


def _openai_tools():
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["schema"]}} for t in tool_specs()]


def _short(result, n=1500):
    text = json.dumps(result, ensure_ascii=False)
    return text if len(text) <= n else text[:n] + " …"


def chat(question, history, state, config, client=None):
    """Answer one question. Returns (answer_text, trace); `state` may be changed by control tools.

    history: previous turns as [{"role": "user"|"assistant", "content": str}].
    client:  optional pre-built SDK client (used by tests).
    """
    if not config.get("api_key") and client is None:
        return ("The AI assistant is not configured: add LLM_API_KEY (and LLM_PROVIDER / LLM_MODEL) "
                "to the app's secrets."), []
    if not config.get("model"):
        return "Set LLM_MODEL in the app's secrets for this provider.", []

    history = [h for h in history if h["role"] in ("user", "assistant")][-2 * HISTORY_TURNS:]
    if config["provider"] == "anthropic":
        return _chat_anthropic(question, history, state, config, client)
    return _chat_openai(question, history, state, config, client)


def _chat_anthropic(question, history, state, config, client):
    if client is None:
        import anthropic
        client = anthropic.Anthropic(api_key=config["api_key"])
    messages = [{"role": h["role"], "content": h["content"]} for h in history]
    messages.append({"role": "user", "content": question})
    tools, trace = _anthropic_tools(), []

    for _ in range(MAX_ROUNDS):
        # no temperature: current Claude models reject sampling parameters
        resp = client.messages.create(model=config["model"], max_tokens=4000,
                                      system=system_prompt(state), tools=tools, messages=messages)
        # pass the blocks back unchanged: thinking blocks must be preserved between tool calls
        messages.append({"role": "assistant", "content": resp.content})

        calls = [b for b in resp.content if b.type == "tool_use"]
        if resp.stop_reason != "tool_use" or not calls:
            return "".join(b.text for b in resp.content if b.type == "text").strip(), trace

        results = []
        for c in calls:
            result = run_tool(c.name, c.input or {}, state)
            trace.append({"tool": c.name, "args": c.input, "result": _short(result)})
            results.append({"type": "tool_result", "tool_use_id": c.id,
                            "content": json.dumps(result, ensure_ascii=False)})
        messages.append({"role": "user", "content": results})

    return "I stopped after too many tool calls without a final answer. Please rephrase the question.", trace


def _chat_openai(question, history, state, config, client):
    if client is None:
        from openai import OpenAI
        client = OpenAI(api_key=config["api_key"], base_url=config.get("base_url") or None)
    messages = [{"role": "system", "content": system_prompt(state)}]
    messages += [{"role": h["role"], "content": h["content"]} for h in history]
    messages.append({"role": "user", "content": question})
    tools, trace = _openai_tools(), []

    for _ in range(MAX_ROUNDS):
        messages[0]["content"] = system_prompt(state)          # state may have changed
        resp = _with_retry(lambda: client.chat.completions.create(model=config["model"], messages=messages, tools=tools))
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return (msg.content or "").strip(), trace

        if hasattr(msg, "model_dump"):
            # keep every field the provider sent (Gemini attaches thought signatures to tool calls)
            messages.append(msg.model_dump(exclude_none=True))
        else:
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [{"id": tc.id, "type": "function",
                                             "function": {"name": tc.function.name,
                                                          "arguments": tc.function.arguments}}
                                            for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = None
            result = (run_tool(tc.function.name, args, state) if isinstance(args, dict)
                      else {"error": "arguments were not valid JSON"})
            trace.append({"tool": tc.function.name, "args": args, "result": _short(result)})
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(result, ensure_ascii=False)})

    return "I stopped after too many tool calls without a final answer. Please rephrase the question.", trace
