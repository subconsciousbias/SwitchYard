#!/usr/bin/env python3
"""Request-surface conformance: is every OpenAI- or Anthropic-compatible
request EXECUTED by every plan -- or refused so it can go elsewhere -- and
never silently degraded? (issue #264)

This spends real quota (a few cheap calls per case per plan), so it is a
script, not a test. Run it after a CLI, LiteLLM or client upgrade, or after
changing a bridge. It runs inside the gateway container (docker compose
exec), where the master key already is -- nothing is printed or copied.

    python3 scripts/request_conformance.py                     # every plan, every case
    python3 scripts/request_conformance.py --plans openai,claude-max
    python3 scripts/request_conformance.py --cases tool_loop_messages,pdf_in_tool_result

Each cell is one of:
  PASS executed  the feature demonstrably worked
  PASS refused   an explicit 4xx/5xx the router or client can act on
  FAIL degraded  a 200 that shows the feature was ignored or made up -- the
                 class web search was in before it was served (#278)
  SKIP capacity  a 429: the plan had no capacity, so the cell proves nothing
  FAIL error     anything else -- including a 404 from a route the plan
                 should serve (LiteLLM's Responses bridge, before #279)

Case shapes follow what real clients send, captured from Claude Code 2.1.281
and OpenCode 1.18.32 against a fake model: Claude Code speaks /v1/messages
with `thinking` + `output_config.effort` and always a max_tokens; OpenCode
speaks /v1/chat/completions with reasoning_effort + max_tokens on every turn.
"""
from __future__ import annotations

import argparse
import json
import subprocess

# Executed INSIDE the gateway container (python3 -), with PLANS/CASES substituted.
PROBE = r'''
import base64, json, os, re, struct, time, urllib.error, urllib.request, zlib
GW = "http://127.0.0.1:4000"
KEY = os.environ["LITELLM_MASTER_KEY"]
TARGETS = %(targets)r
CASES = %(cases)r

def http(path, body, stream=False):
    req = urllib.request.Request(GW + path, data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {KEY}",
        "x-api-key": KEY, "anthropic-version": "2023-06-01"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            raw = r.read().decode()
            return r.status, (raw if stream else json.loads(raw))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:300]

def png(rgb):
    # a solid 16x16 PNG
    raw = b"".join(b"\x00" + bytes(rgb) * 16 for _ in range(16))
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
    return base64.b64encode(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 16, 16, 8, 2, 0, 0, 0))
                            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")).decode()

RED = png((220, 20, 20))
PDF = base64.b64encode(
    b"%%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 400 144]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
    b"4 0 obj<</Length 52>>stream\nBT /F1 24 Tf 20 60 Td (CODEWORD ZEBRA4471) Tj ET\nendstream endobj\n"
    b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\ntrailer<</Root 1 0 R>>\n%%%%EOF").decode()
WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
WEATHER_A = {"name": "get_weather", "description": "Current weather for a city.",
             "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}
READ_A = {"name": "Read", "description": "Read a file from the user's machine.",
          "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}

def chat(model, messages, **kw):
    return http("/v1/chat/completions", {"model": model, "messages": messages, **kw})

def messages_api(model, messages, **kw):
    body = {"model": model, "max_tokens": 4096, "messages": messages}
    body.update(kw)
    return http("/v1/messages", body)

def text_of(resp):
    if isinstance(resp, dict) and "choices" in resp:
        return resp["choices"][0]["message"].get("content") or ""
    if isinstance(resp, dict) and "content" in resp:
        return "".join(b.get("text", "") for b in resp["content"] if isinstance(b, dict))
    return str(resp)

def refused(status):
    return status >= 400

def verdict(status, ok, detail):
    if status == 200 and ok:
        return "PASS executed", detail
    if status == 200:
        return "FAIL degraded", detail
    if status == 429:
        # No capacity right now (quota spent, plan full): says nothing about
        # whether the plan would execute the request.
        return "SKIP capacity", f"HTTP 429: {str(detail)[:120]}"
    if status == 404:
        # A route the plan should serve is missing -- e.g. LiteLLM sending a
        # codex tool turn to /v1/responses, which no sidecar serves.
        return "FAIL error", f"HTTP 404: {str(detail)[:140]}"
    if refused(status):
        return "PASS refused", f"HTTP {status}: {str(detail)[:140]}"
    return "FAIL error", f"HTTP {status}: {str(detail)[:140]}"

# ---- cases -------------------------------------------------------------------
def text_chat(m):
    s, r = chat(m, [{"role": "user", "content": "Reply with exactly the word: pong"}], max_tokens=32000,
                reasoning_effort="medium")
    t = text_of(r) if s == 200 else r
    return verdict(s, s == 200 and "pong" in t.lower(), f"{t[:80]!r}")

def text_messages(m):
    s, r = messages_api(m, [{"role": "user", "content": "Reply with exactly the word: pong"}],
                        max_tokens=64000, thinking={"type": "adaptive"}, output_config={"effort": "medium"})
    t = text_of(r) if s == 200 else r
    return verdict(s, s == 200 and "pong" in t.lower(), f"{t[:80]!r}")

def tool_loop_chat(m):
    # OpenCode's shape: tools + reasoning_effort + max_tokens on every turn.
    msgs = [{"role": "user", "content": "What is the weather in Paris? Use the tool."}]
    s, r = chat(m, msgs, tools=[WEATHER], reasoning_effort="high", max_tokens=32000)
    if s != 200:
        return verdict(s, False, r)
    calls = r["choices"][0]["message"].get("tool_calls") or []
    if not calls or calls[0]["function"]["name"] != "get_weather":
        return verdict(s, False, f"no tool call; answered {text_of(r)[:80]!r}")
    msgs += [r["choices"][0]["message"], {"role": "tool", "tool_call_id": calls[0]["id"],
                                          "content": "Paris: 31C, marker SUNNY-8812"}]
    s, r = chat(m, msgs, tools=[WEATHER], reasoning_effort="high", max_tokens=32000)
    t = text_of(r) if s == 200 else r
    return verdict(s, s == 200 and ("8812" in t or "31" in t), f"final {t[:80]!r}")

def tool_loop_messages(m):
    # Claude Code's shape: /v1/messages + thinking + output_config.effort.
    msgs = [{"role": "user", "content": "What is the weather in Paris? Use the tool."}]
    kw = dict(tools=[WEATHER_A], max_tokens=64000, thinking={"type": "adaptive"}, output_config={"effort": "high"})
    s, r = messages_api(m, msgs, **kw)
    if s != 200:
        return verdict(s, False, r)
    uses = [b for b in r.get("content", []) if b.get("type") == "tool_use"]
    if not uses:
        return verdict(s, False, f"no tool_use; answered {text_of(r)[:80]!r}")
    msgs += [{"role": "assistant", "content": r["content"]},
             {"role": "user", "content": [{"type": "tool_result", "tool_use_id": uses[0]["id"],
                                           "content": "Paris: 31C, marker SUNNY-8812"}]}]
    s, r = messages_api(m, msgs, **kw)
    t = text_of(r) if s == 200 else r
    return verdict(s, s == 200 and ("8812" in t or "31" in t), f"final {t[:80]!r}")

def max_tokens_cap(m):
    s, r = chat(m, [{"role": "user", "content": "Count from 1 to 300, comma-separated, nothing else."}], max_tokens=20)
    if s != 200:
        return verdict(s, False, r)
    t = text_of(r)
    fin = r["choices"][0].get("finish_reason")
    return verdict(s, fin == "length" or len(t) < 200, f"finish={fin} len={len(t)}")

def image_input(m):
    s, r = chat(m, [{"role": "user", "content": [
        {"type": "text", "text": "What single colour fills this image? One word."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{RED}"}}]}])
    t = text_of(r) if s == 200 else r
    return verdict(s, s == 200 and "red" in t.lower(), f"{t[:80]!r}")

def image_in_tool_result(m):
    msgs = [{"role": "user", "content": "Read /tmp/swatch.png and tell me its colour in one word."},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Read",
                                               "input": {"file_path": "/tmp/swatch.png"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": RED}}]}]}]
    s, r = messages_api(m, msgs, tools=[READ_A])
    t = text_of(r) if s == 200 else r
    return verdict(s, s == 200 and "red" in t.lower(), f"{t[:80]!r}")

def pdf_in_tool_result(m):
    msgs = [{"role": "user", "content": "Read /tmp/doc.pdf and tell me the codeword in it."},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Read",
                                               "input": {"file_path": "/tmp/doc.pdf"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": PDF}}]}]}]
    s, r = messages_api(m, msgs, tools=[READ_A])
    t = text_of(r) if s == 200 else r
    return verdict(s, s == 200 and "4471" in t, f"{t[:80]!r}")

# "I can't browse", "search is unavailable", "as of my knowledge cutoff"...: a
# 200 that says it did not search is degraded even when it invents a URL.
CANNOT = re.compile(
    r"(?:\b(?:no|not|don'?t|doesn'?t|isn'?t|do\s+not|does\s+not|cannot|can'?t|unable\s+to|lack|without)\b[^.\n]{0,40}"
    r"\b(?:search|brows|web|internet|online|fetch|real-?time|live)"
    r"|\b(?:search|brows\w*|web\s+access|real-?time)[^.\n]{0,30}\b(?:unavailable|not\s+(?:available|supported|possible))"
    r"|knowledge\s+cut-?off|training\s+data)", re.I)
QUOTED = re.compile(r'"[^"\n]*"|“[^”\n]*”')

def cannot(text):
    # Quoted titles ("Show HN: Not another web framework") are the answer, not a refusal.
    return bool(CANNOT.search(QUOTED.sub("", text)))

WEB_PROMPT = ("Search the web: what is the title of the current top story on "
              "news.ycombinator.com? Quote it and give the source URL.")

def web_verdict(s, r):
    t = text_of(r) if s == 200 else r
    # Executed means the answer cites the site it was sent to and does not
    # say it could not look -- not merely that some URL appears.
    ok = s == 200 and "ycombinator.com" in t and not cannot(t)
    return verdict(s, ok, f"{t[:100]!r}")

def web_search_requested(m):
    # What LiteLLM makes of Claude Code's WebSearch sub-request, sent as an
    # OpenAI-shaped caller would.
    s, r = chat(m, [{"role": "user", "content": WEB_PROMPT}], web_search_options={})
    return web_verdict(s, r)

def web_search_messages(m):
    # Claude Code's WebSearch sub-request as captured (client_shapes.json):
    # /v1/messages with Anthropic's server-side web_search tool.
    s, r = messages_api(m, [{"role": "user", "content": WEB_PROMPT}],
                        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 1}])
    return web_verdict(s, r)

def web_fetch_messages(m):
    # Anthropic's server-side web_fetch tool: one URL, a known page.
    s, r = messages_api(m, [{"role": "user", "content": "Fetch https://example.com and quote its main "
                                                        "heading exactly."}],
                        tools=[{"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 1}])
    t = text_of(r) if s == 200 else r
    return verdict(s, s == 200 and "example domain" in t.lower() and not cannot(t), f"{t[:100]!r}")

def structured_output(m):
    s, r = chat(m, [{"role": "user", "content": "What is 2+3?"}], response_format={
        "type": "json_schema", "json_schema": {"name": "sum", "strict": True, "schema": {
            "type": "object", "properties": {"answer": {"type": "integer"}},
            "required": ["answer"], "additionalProperties": False}}})
    if s != 200:
        return verdict(s, False, r)
    t = text_of(r).strip()
    try:
        ok = json.loads(t) == {"answer": 5}
    except ValueError:
        ok = False
    return verdict(s, ok, f"{t[:80]!r}")

def n_choices(m):
    s, r = chat(m, [{"role": "user", "content": "Say hi."}], n=2)
    if s != 200:
        return verdict(s, False, r)
    return verdict(s, len(r.get("choices", [])) == 2, f"{len(r.get('choices', []))} choice(s)")

def stop_sequence(m):
    s, r = chat(m, [{"role": "user", "content": "Repeat exactly: alpha beta STOPHERE gamma delta"}], stop=["STOPHERE"])
    t = text_of(r) if s == 200 else r
    return verdict(s, s == 200 and "gamma" not in t and "alpha" in t, f"{t[:80]!r}")

def streamed_tool_call(m):
    s, raw = http("/v1/chat/completions", {"model": m, "stream": True, "tools": [WEATHER],
        "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}]}, stream=True)
    ok = s == 200 and "get_weather" in raw and "tool_calls" in raw
    return verdict(s, ok, "tool_call in stream" if ok else (raw[:120] if isinstance(raw, str) else raw))

def unsupported_server_tool(m):
    # Anthropic's server-side code execution: no CLI or chat backend runs it
    # here. Executed would need a real sandbox; the honest outcomes are a
    # real result or an explicit refusal.
    s, r = messages_api(m, [{"role": "user", "content": "Use code execution to compute 123456789*987654321 and give the exact product."}],
                        tools=[{"type": "code_execution_20250825", "name": "code_execution"}])
    t = text_of(r) if s == 200 else r
    return verdict(s, s == 200 and "121932631112635269" in t.replace(",", ""), f"{str(t)[:100]!r}")

def _responses_tool_call(r):
    if not isinstance(r, dict):
        return None
    calls = [o for o in r.get("output", []) if o.get("type") == "function_call"]
    return calls[0] if calls else None

def responses_standard_tools(m):
    # The Responses API as an OpenAI SDK sends it: top-level function tools.
    s, r = http("/v1/responses", {"model": m, "reasoning": {"effort": "medium"},
        "input": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
        "tools": [{"type": "function", "name": "get_weather", "description": "Current weather for a city.",
                   "parameters": WEATHER["function"]["parameters"]}]})
    call = _responses_tool_call(r) if s == 200 else None
    return verdict(s, bool(call) and call.get("name") == "get_weather",
                   f"call {call.get('name') if call else None}" if s == 200 else r)

def responses_codex_client_shape(m):
    # Codex CLI as a client: tools inside an `additional_tools` input item,
    # namespaced, including a freeform `custom` grammar tool (captured from
    # codex-cli 0.153.4 by scripts/capture_client_shapes.py).
    s, r = http("/v1/responses", {"model": m, "reasoning": {"effort": "medium"}, "stream": False,
        "input": [
            {"type": "additional_tools", "role": "developer", "tools": [{"type": "namespace", "name": "functions",
                "tools": [{"type": "function", "name": "get_weather", "description": "Current weather for a city.",
                           "parameters": WEATHER["function"]["parameters"], "strict": False},
                          {"type": "custom", "name": "apply_patch", "description": "Edit files with a patch.",
                           "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"}}]}]},
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "What is the weather in Paris? Use the tool."}]}]})
    call = _responses_tool_call(r) if s == 200 else None
    return verdict(s, bool(call) and call.get("name") == "get_weather",
                   f"call {call.get('name') if call else None}; output types "
                   f"{[o.get('type') for o in r.get('output', [])] if isinstance(r, dict) else ''}" if s == 200 else r)

ALL = [text_chat, text_messages, tool_loop_chat, tool_loop_messages, max_tokens_cap, image_input,
       image_in_tool_result, pdf_in_tool_result, web_search_requested, web_search_messages,
       web_fetch_messages, structured_output, n_choices,
       stop_sequence, streamed_tool_call, unsupported_server_tool,
       responses_standard_tools, responses_codex_client_shape]
for model in TARGETS:
    for case in ALL:
        if CASES and case.__name__ not in CASES:
            continue
        started = time.time()
        try:
            status, detail = case(model)
        except Exception as exc:
            status, detail = "FAIL error", f"{type(exc).__name__}: {exc}"
        print(json.dumps({"model": model, "case": case.__name__, "status": status,
                          "detail": str(detail)[:200], "seconds": round(time.time() - started, 1)}), flush=True)
'''

LIST = r'''
import json, yaml
d = yaml.safe_load(open("/tmp/litellm.generated.yaml"))
seen = {}
for m in d["model_list"]:
    info = m.get("model_info") or {}
    plan = info.get("switchyard_plan")
    if plan and info.get("enabled", True) and plan not in seen:
        seen[plan] = {"model": m["model_name"], "auth": info.get("auth")}
print(json.dumps(seen))
'''


def compose(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", "compose", *args], input=stdin, capture_output=True,
                          text=True, timeout=7200)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plans", help="comma-separated plan keys (default: every enabled plan)")
    ap.add_argument("--cases", help="comma-separated case names (default: all)")
    args = ap.parse_args()
    listed = compose("exec", "-T", "gateway", "python3", "-c", LIST)
    if listed.returncode != 0:
        print(f"cannot read the gateway's plans: {listed.stderr.strip()[:300]}")
        return 2
    plans = json.loads(listed.stdout.strip().splitlines()[-1])
    wanted = set(args.plans.split(",")) if args.plans else set(plans)
    targets = [info["model"] for plan, info in plans.items() if plan in wanted]
    cases = args.cases.split(",") if args.cases else []
    kinds = {info["model"]: f"{plan} ({info['auth']})" for plan, info in plans.items()}
    probe = PROBE % {"targets": targets, "cases": cases}
    proc = subprocess.Popen(["docker", "compose", "exec", "-T", "gateway", "python3", "-"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    proc.stdin.write(probe)
    proc.stdin.close()
    failed = 0
    current = None
    for line in proc.stdout:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row["model"] != current:
            current = row["model"]
            print(f"\n== {kinds.get(current, current)}  [{current}]", flush=True)
        failed += row["status"].startswith("FAIL")
        print(f"   {row['status']:<14} {row['case']:<24} {row['detail']}", flush=True)
    proc.wait()
    if proc.returncode != 0:
        print(f"probe error: {proc.stderr.read().strip()[-400:]}")
        failed += 1
    print(f"\n{'every cell executed or explicitly refused' if not failed else f'{failed} cell(s) failed'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
