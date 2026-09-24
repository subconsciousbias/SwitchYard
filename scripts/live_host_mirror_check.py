#!/usr/bin/env python3
"""Live host-mirror check: what does each RUNNING sidecar's CLI really give
the model, on its real login? (issue #264)

This spends real quota -- a handful of cheap calls per sidecar -- which is
why it is a script and not a test. Run it when a pinned CLI version changes
(after `scripts/apply.sh --build`), or whenever the tool path misbehaves.

    python3 scripts/live_host_mirror_check.py                 # every mcp sidecar
    python3 scripts/live_host_mirror_check.py codex-sidecar   # just one

Why it exists next to the offline guards. tests/test_lockdown_*.py (CI) and
the startup self-check (sidecars/mcp_bridge/selfcheck.py) drive the pinned
CLI against a loopback fake model, so they see exactly what the CLI sends --
but never what the REAL backend adds. The first live run found one: on a
ChatGPT login codex's model had a working `web.run` search tool that no
fake-model check could observe. This script asks the real model, through the
sidecar's own /v1/chat/completions (the production bridge path), to:

positive -- the caller's world is what the model sees and uses:
  loop         run `echo host-mirror-ok` with the caller's bridged `bash`; the
               first action must be the bridged tool and the loop must finish
  tools        list every tool it has: all three caller tools must be there
               (bash, read_file, a Claude Code-style mcp__github__get_issue),
               and nothing that looks like a web/shell/agent/question tool
  MCP tool     use the caller's GitHub MCP tool, with the right arguments
  list cwd     list the working directory through the bridged tool, never
               naming a relay path, and report the host's files
  read file    read README.md from the caller (host-only marker content)
  environment  without tools, name the OS (macOS), the host cwd, git = yes
  environment (Windows caller)
               the same for a Claude Code-style Windows caller (win32, a
               C:\\ cwd, a PowerShell tool): Windows, that exact cwd, git = yes
  web search   a request carrying web_search_options (Claude Code's WebSearch
               arrives this way) gets a live, sourced answer from the CLI's own
               provider-side search -- text path and tool path
negative -- nothing of the relay's own is reachable:
  web          without that request, a live web answer instead of NO WEB TOOL
               fails (tool path and text path)
  shell        output from the sidecar's own user (uid=1000/node) fails

It runs inside each sidecar via `docker compose exec` and talks to it on
loopback, so it needs nothing but the running stack: no gateway key, no
change to anything. Exits non-zero if any check fails.
"""
from __future__ import annotations

import json
import subprocess
import sys

# Executed INSIDE the sidecar (python3 -), with PORT and MODEL substituted.
PROBE = r'''
import json, re, time, urllib.request
URL = "http://127.0.0.1:%(port)s/v1/chat/completions"
MODEL = %(model)r
HOST_CWD = "/Users/switchyard-live-check/project"
SYSTEM = ("You are a coding agent working on the caller's machine.\n<env>\n"
          f"Working directory: {HOST_CWD}\nIs directory a git repo: yes\nPlatform: darwin\n</env>")

def tool(name, description, props, required):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": props, "required": required}}}

BASH = tool("bash", "Run a shell command on the caller's machine and return its output.",
            {"command": {"type": "string"}}, ["command"])
READ = tool("read_file", "Read a text file on the caller's machine.",
            {"path": {"type": "string", "description": "Absolute or cwd-relative path"}}, ["path"])
# A client-side MCP tool, named the way Claude Code exposes one (mcp__<server>__<tool>).
GH = tool("mcp__github__get_issue", "Fetch a GitHub issue (GitHub MCP server on the caller).",
          {"owner": {"type": "string"}, "repo": {"type": "string"},
           "issue_number": {"type": "integer"}}, ["owner", "repo", "issue_number"])
CALLER_TOOLS = [BASH, READ, GH]
HOST_ID = "uid=501(caller) gid=20(staff)"
LISTING = "README.md\npyproject.toml\nsrc\n"
README = "# Caller Project Marker\nThis README exists only on the caller.\n"
RELAY_PATH = re.compile(r"/tmp/|/app/|/home/node|/relay/|sy-cli-|mcpb-")

def post(body):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))

def host_answer(name, args):
    # What the caller's own tools would return.
    if name == "read_file":
        return README if "README" in str(args.get("path", "")) else "error: no such file\n"
    if name == "mcp__github__get_issue":
        return json.dumps({"number": args.get("issue_number"),
                           "title": "Host-mirror tool calling (live check marker)"})
    cmd = str(args.get("command", ""))
    if "host-mirror-ok" in cmd:
        return "host-mirror-ok\n"
    if re.search(r"\bls\b|\bdir\b|find ", cmd):
        return LISTING
    if "README" in cmd and ("cat" in cmd or "head" in cmd):
        return README
    if re.search(r"\bid\b", cmd):
        return HOST_ID + "\n"
    if "pwd" in cmd or "uname" in cmd:
        return f"cwd={HOST_CWD}\nplatform=Darwin\nshell=/bin/zsh\n"
    return "\n"

def run(user, tools=None, turns=6, system=SYSTEM):
    # Drive one conversation, answering every bridged call as the host would.
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    calls = []
    for _ in range(turns):
        body = {"model": MODEL, "messages": messages}
        if tools:
            body["tools"] = tools
        msg = post(body)["choices"][0]["message"]
        if not msg.get("tool_calls"):
            return (msg.get("content") or ""), calls
        messages.append(msg)
        for call in msg["tool_calls"]:
            args = json.loads(call["function"]["arguments"] or "{}")
            calls.append((call["function"]["name"], args))
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": host_answer(call["function"]["name"], args)})
    return "(no final answer after %%d turns)" %% turns, calls

def emit(name, ok, detail, started=None):
    out = {"check": name, "ok": ok, "detail": detail}
    if started is not None:
        out["seconds"] = round(time.time() - started, 1)
    print(json.dumps(out), flush=True)

def check(name, fn):
    started = time.time()
    try:
        ok, detail = fn()
    except Exception as exc:
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    emit(name, ok, detail, started)

def short(text):
    return repr(text.strip().replace("\n", " | ")[:170])

# ---- positive: the caller's world is what the model sees and uses ----------
def loop():
    text, calls = run("Use the bash tool to run `echo host-mirror-ok`. After you get the "
                      "result, reply with the output on one line, then list the exact name "
                      "of every tool you have available, one per line, prefixed 'TOOL: '.",
                      CALLER_TOOLS)
    listed = re.findall(r"TOOL:\s*(\S+)", text)
    missing = [t for t in ("bash", "read_file", "mcp__github__get_issue")
               if not any(t in name for name in listed)]
    suspicious = [t for t in listed if re.search(
        r"web|search|fetch|browse|shell|exec|terminal|agent|task|question|ask|patch|edit|write",
        t, re.I) and not any(c in t for c in ("bash", "read_file", "github"))]
    emit("tools listed", not missing and not suspicious, f"model lists {listed}"
         + (f"; MISSING caller tools {missing}" if missing else "")
         + (f"; SUSPICIOUS {suspicious}" if suspicious else ""))
    first = calls[0] if calls else None
    ok = bool(first) and first[0] == "bash" and "host-mirror-ok" in text
    return ok, f"first action {first}; answer {short(text.splitlines()[0] if text else '')}"

def mcp_tool():
    text, calls = run("What is the title of issue 264 in the GitHub repo Fledgewing/SwitchYard? "
                      "Use your GitHub tool.", CALLER_TOOLS)
    gh = [a for n, a in calls if n == "mcp__github__get_issue"]
    ok = bool(gh) and int(gh[0].get("issue_number", 0)) == 264 and "Host-mirror" in text
    return ok, f"calls {calls}; answer {short(text)}"

def listing():
    text, calls = run("List the files in the current working directory.", CALLER_TOOLS)
    cmds = [a.get("command", "") for n, a in calls if n == "bash"]
    relay = [c for c in cmds if RELAY_PATH.search(c)]
    ok = bool(cmds) and not relay and "pyproject.toml" in text
    return ok, f"bash commands {cmds}" + (f"; RELAY PATHS {relay}" if relay else "") + f"; answer {short(text)}"

def read():
    text, calls = run("Read README.md in the working directory and tell me its first line.",
                      CALLER_TOOLS)
    paths = [str(a.get("path") or a.get("command")) for n, a in calls if n in ("read_file", "bash")]
    relay = [p for p in paths if RELAY_PATH.search(p)]
    ok = bool(paths) and not relay and "Caller Project Marker" in text
    return ok, f"calls {calls}" + (f"; RELAY PATHS {relay}" if relay else "") + f"; answer {short(text)}"

ENV_Q = ("Without calling any tool, answer on exactly three lines: 1) the operating "
         "system of the machine your tools run on, 2) the absolute path of the current "
         "working directory, 3) whether that directory is a git repository (yes/no).")

def environment():
    text, calls = run(ENV_Q, CALLER_TOOLS)
    low = text.lower()
    os_ok = ("mac" in low or "darwin" in low) and not re.search(r"\blinux\b", low.split("\n")[0])
    ok = os_ok and HOST_CWD in text and "yes" in low and not RELAY_PATH.search(text)
    return ok, f"answer {short(text)}; tool calls {calls}"

# A Windows caller, as Claude Code on Windows describes itself: its tools run
# in PowerShell on C:\..., which no relay directory can mirror, so only the
# env block and reminder stand between the model and the relay's Linux cwd.
WIN_CWD = r"C:\Users\switchyard-live-check\project"
WIN_SYSTEM = ("You are a coding agent working on the caller's machine.\n# Environment\n"
              f" - Primary working directory: {WIN_CWD}\n - Is a git repository: true\n"
              " - Platform: win32\n - Shell: PowerShell\n - OS Version: Windows 11 Pro\n")
POWERSHELL = tool("PowerShell", "Run a PowerShell command on the caller's machine.",
                  {"command": {"type": "string"}}, ["command"])

def environment_windows():
    text, calls = run(ENV_Q, [POWERSHELL, READ], system=WIN_SYSTEM)
    low = text.lower()
    first = low.split("\n")[0]
    os_ok = "windows" in first and not re.search(r"\blinux\b", first)
    ok = os_ok and WIN_CWD.lower() in low and "yes" in low and not RELAY_PATH.search(text)
    return ok, f"answer {short(text)}; tool calls {calls}"

# ---- negative: nothing of the relay's own is reachable ----------------------
WEB_Q = ("Do not use bash. Use a web search or browsing tool, if you really have one, to find the "
         "title of the current top story on news.ycombinator.com and quote it. If you have no "
         "working web tool, reply exactly NO WEB TOOL and nothing else.")

def web(tools):
    def _run():
        text, calls = run(WEB_Q, tools)
        return "NO WEB TOOL" in text.upper(), f"answer {short(text)}; bridged calls {calls}"
    return _run

def shell():
    text, calls = run("Do NOT use the bash tool. Run the command `id` with any built-in shell, "
                      "terminal or exec tool of your own and report its exact output. If you "
                      "have no such tool, reply exactly NO NATIVE SHELL.", CALLER_TOOLS)
    ran_in_sidecar = "uid=1000" in text or "(node)" in text
    return not ran_in_sidecar, f"answer {short(text)}; bridged calls {calls}"

check("loop via bridged bash", loop)
check("client MCP tool", mcp_tool)
check("list cwd", listing)
check("read file", read)
check("environment (os/cwd/git)", environment)
check("environment, Windows caller (os/cwd/git)", environment_windows)
def web_requested(tools):
    # A caller that ASKS for server-side web search (web_search_options, what
    # LiteLLM makes of Claude Code's WebSearch sub-request) gets the CLI's own
    # provider-side search: a live answer with a source URL.
    def _run():
        messages = [{"role": "user", "content": (
            "Search the web for the title of the current top story on "
            "news.ycombinator.com; quote it with the source URL. If you cannot "
            "search, reply exactly NO WEB TOOL.")}]
        body = {"model": MODEL, "messages": messages, "web_search_options": {}}
        if tools:
            body["tools"] = tools
        msg = post(body)["choices"][0]["message"]
        text = msg.get("content") or ""
        ok = "NO WEB TOOL" not in text.upper() and "http" in text
        return ok, f"answer {short(text)}"
    return _run

check("web search when requested (text path)", web_requested(None))
check("web search when requested (tool path)", web_requested(CALLER_TOOLS))
check("no web (tool path)", web(CALLER_TOOLS))
check("no web (text path)", web(None))
check("no native shell", shell)
'''

IDENTITY = ("import json,os,urllib.request;"
            "h=json.load(urllib.request.urlopen('http://127.0.0.1:'+os.environ['SIDECAR_PORT']+'/health',timeout=10));"
            "print(json.dumps({k:os.environ.get(k) for k in ('BRIDGE','SIDECAR_PORT','PROVIDER')}"
            "|{'model':h.get('model'),'host_mirror':h.get('host_mirror')}))")


def compose(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", "compose", *args], input=stdin, capture_output=True,
                          text=True, timeout=3600)


def sidecars(wanted: list[str]) -> list[str]:
    running = compose("ps", "--services", "--status", "running").stdout.split()
    return [s for s in running if s.endswith("-sidecar") and (not wanted or s in wanted)]


def main(argv: list[str]) -> int:
    failed = 0
    services = sidecars(argv)
    if not services:
        print("no running sidecar matched; is the stack up (run from the main checkout)?")
        return 2
    for service in services:
        ident = compose("exec", "-T", service, "python3", "-c", IDENTITY)
        if ident.returncode != 0:
            print(f"\n== {service}: cannot read its health ({ident.stderr.strip()[:200]})")
            failed += 1
            continue
        info = json.loads(ident.stdout.strip().splitlines()[-1])
        mirror = info.get("host_mirror") or {}
        print(f"\n== {service}  provider={info['PROVIDER']} bridge={info['BRIDGE']} "
              f"model={info['model']}  self-check={mirror.get('status', 'n/a')}", flush=True)
        if info["BRIDGE"] != "mcp":
            print("   skipped: not a tool-path (mcp) sidecar")
            continue
        probe = PROBE % {"port": info["SIDECAR_PORT"], "model": info["model"]}
        done = compose("exec", "-T", service, "python3", "-", stdin=probe)
        for line in done.stdout.splitlines():
            try:
                result = json.loads(line)
            except ValueError:
                continue
            mark = "PASS" if result["ok"] else "FAIL"
            failed += not result["ok"]
            seconds = f" ({result['seconds']}s)" if "seconds" in result else ""
            print(f"   {mark}  {result['check']}{seconds}: {result['detail']}", flush=True)
        if done.returncode != 0:
            print(f"   probe error: {done.stderr.strip()[-400:]}")
            failed += 1
    print(f"\n{'all checks passed' if not failed else f'{failed} check(s) failed'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
