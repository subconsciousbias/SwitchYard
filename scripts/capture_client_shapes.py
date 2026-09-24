#!/usr/bin/env python3
"""What do the CLIENTS send? Capture each locally installed client's request
shape and diff it against the committed baseline (issue #264).

A client upgrade changes what reaches SwitchYard the same way a CLI upgrade
changes what serves it: a new top-level parameter (Claude Code's
`output_config.effort`), a new server-side tool type (its WebSearch
sub-request's `web_search_20250305`), a new content block. Any of those can
be silently dropped somewhere between the client and the model. This runs
the clients installed on THIS machine -- claude, opencode, codex -- against a
loopback fake model (tests/_fake_model.py; no network, no credentials, no
quota), records only the SHAPE of every request (keys, tool types and fields,
content-block types, header names -- never values), and diffs it against
tests/fixtures/client_shapes.json.

    python3 scripts/capture_client_shapes.py            # diff against the baseline
    python3 scripts/capture_client_shapes.py --update   # accept the current shapes

A difference is not an error in itself: it is the list of things to check
(`scripts/request_conformance.py`) before trusting the upgraded client
through SwitchYard. Exits 1 when the shapes differ from the baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
import _fake_model  # noqa: E402

BASELINE = ROOT / "tests" / "fixtures" / "client_shapes.json"
PROMPT = "Search the web for the latest Python release, then read README.md."
# Script: the first tool-bearing request gets a call to the client's own web
# tool, so any server-side sub-request it makes to execute it is captured too.
SCRIPTS = {
    "claude": [{"tool": "WebSearch", "input": {"query": "latest python release"}}],
    "opencode": [{"tool": "webfetch", "input": {"url": "https://www.python.org", "format": "text"}}],
    "codex": [],
}


def block_types(content) -> list[str]:
    if isinstance(content, list):
        return sorted({str(b.get("type")) for b in content if isinstance(b, dict)})
    return ["string"] if isinstance(content, str) else []


def shape(request: dict) -> dict:
    """The value-free signature of one captured request."""
    body = request["body"]
    tools = list(body.get("tools") or [])
    for item in body.get("input") or [] if isinstance(body.get("input"), list) else []:
        if isinstance(item, dict) and item.get("type") == "additional_tools":
            for tool in item.get("tools") or []:
                tools += tool.get("tools") or [tool] if tool.get("type") == "namespace" else [tool]
    messages = body.get("messages") or body.get("input") or []
    return {
        "path": request["path"],
        "params": sorted(k for k in body if k not in ("messages", "input", "tools", "system")),
        "nested_params": sorted(f"{k}.{sub}" for k in ("output_config", "thinking", "reasoning",
                                                       "stream_options", "context_management")
                                if isinstance(body.get(k), dict) for sub in body[k]),
        "tool_types": sorted({str(t.get("type") or "(plain)") for t in tools if isinstance(t, dict)}),
        "tool_fields": sorted({f for t in tools if isinstance(t, dict)
                               for f in (t.get("function") or t)}),
        "message_block_types": sorted({bt for m in messages if isinstance(m, dict)
                                       for bt in block_types(m.get("content"))}
                                      | {str(m.get("type")) for m in messages
                                         if isinstance(m, dict) and m.get("type")}),
        "headers": sorted(h.lower() for h in request["headers"]
                          if h.lower().startswith(("anthropic-", "x-", "openai-"))
                          and "key" not in h.lower() and "auth" not in h.lower()),
    }


def run_client(client: str, fake: _fake_model.FakeModel, home: Path, cwd: Path) -> None:
    env = {"PATH": os.environ["PATH"], "HOME": str(home), "TERM": "dumb"}
    if client == "claude":
        env.update(ANTHROPIC_BASE_URL=fake.url, ANTHROPIC_API_KEY="shape-capture",
                   CLAUDE_CONFIG_DIR=str(home / "claude"), CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1")
        argv = ["claude", "-p", PROMPT, "--output-format", "json",
                "--allowed-tools", "WebSearch,WebFetch,Read"]
    elif client == "opencode":
        config = home / "config" / "opencode"
        config.mkdir(parents=True)
        (config / "opencode.json").write_text(json.dumps({"provider": {
            "shape-openai": {"npm": "@ai-sdk/openai-compatible", "name": "shape-openai",
                             "options": {"baseURL": f"{fake.url}/v1", "apiKey": "x"},
                             "models": {"m": {"name": "m", "tool_call": True, "reasoning": True}}},
            "shape-anthropic": {"npm": "@ai-sdk/anthropic", "name": "shape-anthropic",
                                "options": {"baseURL": f"{fake.url}/v1", "apiKey": "x"},
                                "models": {"claude-opus-5-5": {"name": "claude-opus-5-5",
                                                               "tool_call": True, "reasoning": True}}}}}))
        env.update(XDG_CONFIG_HOME=str(home / "config"), XDG_DATA_HOME=str(home / "data"),
                   XDG_STATE_HOME=str(home / "state"), XDG_CACHE_HOME=str(home / "cache"))
        for model in ("shape-openai/m", "shape-anthropic/claude-opus-5-5"):
            subprocess.run(["opencode", "run", "--model", model, "--format", "json", "--", PROMPT],
                           env=env, cwd=cwd, capture_output=True, timeout=180,
                           stdin=subprocess.DEVNULL)
        return
    else:
        (home / "codex").mkdir()
        env.update(CODEX_HOME=str(home / "codex"), SHAPE_KEY="x")
        argv = ["codex", "exec", "--skip-git-repo-check", "-m", "gpt-5.6-terra",
                "-c", "model_provider=shape",
                "-c", f'model_providers.shape={{name="shape",base_url="{fake.url}/v1",'
                      'env_key="SHAPE_KEY",wire_api="responses"}', PROMPT]
    subprocess.run(argv, env=env, cwd=cwd, capture_output=True, timeout=180,
                   stdin=subprocess.DEVNULL)


def capture() -> dict:
    shapes: dict = {}
    for client in ("claude", "opencode", "codex"):
        path = shutil.which(client)
        if not path:
            print(f"  {client}: not installed here, skipped")
            continue
        out = subprocess.run([client, "--version"], capture_output=True, text=True, timeout=60)
        # Some builds print the version on stderr; none at all is recorded as
        # such (a diff line), not a crash that skips the remaining clients.
        lines = (out.stdout.strip() or out.stderr.strip()).splitlines()
        version = lines[0] if lines else "(unknown)"
        home = Path(tempfile.mkdtemp(prefix=f"shape-{client}-"))
        cwd = home / "project"
        cwd.mkdir()
        subprocess.run(["git", "init", "-q", str(cwd)], capture_output=True)
        (cwd / "README.md").write_text("# project\n")
        try:
            with _fake_model.FakeModel(SCRIPTS[client]) as fake:
                run_client(client, fake, home, cwd)
            seen = {}
            for request in fake.requests:
                sig = shape(request)
                seen[json.dumps(sig, sort_keys=True)] = sig
            shapes[client] = {"version": version,
                              "requests": sorted(seen.values(), key=lambda s: json.dumps(s, sort_keys=True))}
            print(f"  {client} {version}: {len(fake.requests)} request(s), {len(seen)} distinct shape(s)")
        finally:
            shutil.rmtree(home, ignore_errors=True)
    return shapes


def diff(old: dict, new: dict) -> list[str]:
    out = []
    for client in sorted(set(old) | set(new)):
        before = {json.dumps(s, sort_keys=True) for s in (old.get(client) or {}).get("requests", [])}
        after = {json.dumps(s, sort_keys=True) for s in (new.get(client) or {}).get("requests", [])}
        if client in new and client in old and old[client]["version"] != new[client]["version"]:
            out.append(f"{client}: version {old[client]['version']} -> {new[client]['version']}")
        out += [f"{client}: NEW shape {s}" for s in sorted(after - before)]
        out += [f"{client}: GONE shape {s}" for s in sorted(before - after)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--update", action="store_true", help="write the current shapes as the baseline")
    args = ap.parse_args()
    shapes = capture()
    if args.update:
        BASELINE.write_text(json.dumps(shapes, indent=2, sort_keys=True) + "\n")
        print(f"baseline written: {BASELINE.relative_to(ROOT)}")
        return 0
    old = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
    changes = diff({k: v for k, v in old.items() if k in shapes}, shapes)
    for line in changes:
        print(line)
    print("client request shapes match the baseline" if not changes
          else f"{len(changes)} difference(s): check them with scripts/request_conformance.py")
    return 1 if changes else 0


if __name__ == "__main__":
    raise SystemExit(main())
