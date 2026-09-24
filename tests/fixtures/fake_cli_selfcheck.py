#!/usr/bin/env python3
"""A stand-in CLI for tests/test_selfcheck.py. Speaks just enough of the
Anthropic /v1/messages protocol to the self-check's fake model, and behaves
as the sibling `.mode` file says:
  obedient -- offers only the bridged probe tool; ignores the native call
  leaky    -- also offers `Bash`, and RUNS the native call it is given
  hides    -- offers no tool at all (the bridged tool is hidden)
"""
import json
import os
import pathlib
import subprocess
import urllib.request

mode = pathlib.Path(__file__).with_name("fake_cli_selfcheck.mode").read_text().strip()
url = os.environ["ANTHROPIC_BASE_URL"]
tools = [] if mode == "hides" else [{"name": "mcp__switchyard__switchyard_selfcheck_probe"}]
if mode == "leaky":
    tools.append({"name": "Bash"})


def post(body):
    req = urllib.request.Request(url + "/v1/messages", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=10).read().decode()


out = post({"model": "m", "stream": True, "tools": tools, "messages": []})
events = [json.loads(line[6:]) for line in out.splitlines() if line.startswith("data: ")]
args = "".join(e["delta"]["partial_json"] for e in events
               if e.get("type") == "content_block_delta"
               and e["delta"].get("type") == "input_json_delta")
if args and mode == "leaky":
    subprocess.run(json.loads(args)["command"], shell=True, check=False, cwd=os.getcwd())
post({"model": "m", "stream": True, "tools": tools,
      "messages": [{"role": "user", "content": "continue"}]})
print(json.dumps({"type": "result", "subtype": "success", "result": "done"}))
