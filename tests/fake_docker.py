#!/usr/bin/env python3
"""A stateful fake `docker` CLI for the offline apply.sh / image_plan.py tests.

Copied onto PATH as `docker` by `install()`. It never talks to a daemon: every
answer comes from a JSON state file ($FAKE_DOCKER_STATE), and every invocation
is appended, one line of argv, to $FAKE_DOCKER_LOG. Mutating commands (build,
up, stop) update the state, so a test can run image_plan.py -> build -> up and
then re-plan against the result.

State shape (all keys optional except `services`):

    {
      "project": "sy",
      "services": {
        "gateway": {"build": {"dockerfile": "Dockerfile.gateway"},
                    "environment": {"K": "v"}},
        "codex-sidecar": {"image": "switchyard-sidecar:latest",
                          "environment": {"SWITCHYARD_PLAN": "openai"}}
      },
      "images":     {"sy-gateway:latest": {"Id": "sha256:1", "Labels": {}}},
      "containers": {"gateway": {"ID": "c-gateway", "Image": "sha256:1",
                                 "Env": ["K=v"], "Status": "running",
                                 "Health": "healthy"}},
      "zcard":  {"openai": [2, 0]},        # successive ZCARD answers; the last repeats
      "health": {"codex-sidecar": ["busy", "idle"]},   # health_idle.py answers
      "health_after": {"codex-sidecar": "unhealthy"},  # Health after a recreate
      "preflight_fail": ["codex-sidecar"],
      "redis_get": {"switchyard:router_sig": "abc"},   # redis-cli GET answers
      "logs": ["", "gateway reloaded in place"],       # successive `compose logs`
      "ports": {"gateway": "127.0.0.1:4555"}           # `compose port` answers
    }

A container row may carry "Service" when its key is not the service name
(two containers of one service). Image rows may carry "Env", the image's own
ENV, which a recreated container inherits under the compose environment.

Concurrent invocations (apply.sh runs drains in background jobs) are
serialised with an fcntl lock on the state file.
"""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import stat
import sys


FAKE_CURL = """#!/bin/sh
# Fake curl: the gateway's liveliness probe succeeds, everything else (the
# portal's /healthz and /admin/reload) is unreachable -- so no test can ever
# reach a live stack on localhost. Each call is logged as "curl ARGS".
[ -n "$FAKE_DOCKER_LOG" ] && echo "curl $*" >> "$FAKE_DOCKER_LOG"
for a in "$@"; do
  case "$a" in *health/liveliness*) exit 0 ;; esac
done
exit 7
"""


def install(bin_dir: str, curl: bool = False) -> str:
    """Copy this script to <bin_dir>/docker (executable) and return its path.
    With curl=True also install a fake `curl` next to it."""
    os.makedirs(bin_dir, exist_ok=True)
    dest = os.path.join(bin_dir, "docker")
    shutil.copy2(os.path.abspath(__file__), dest)
    os.chmod(dest, os.stat(dest).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    if curl:
        c = os.path.join(bin_dir, "curl")
        with open(c, "w") as fh:
            fh.write(FAKE_CURL)
        os.chmod(c, 0o755)
    return dest


def _tag(project: str, svc: str, spec: dict) -> str:
    image = spec.get("image") or f"{project}-{svc}"
    last = image.rsplit("/", 1)[-1]
    return image if ":" in last else image + ":latest"


def _pop(seq_map: dict, key: str, default):
    seq = seq_map.get(key)
    if seq is None:
        return default
    if not isinstance(seq, list):
        return seq
    if len(seq) > 1:
        return seq.pop(0)
    return seq[0] if seq else default


def _env_list(spec: dict, image_env: list[str] | None = None) -> list[str]:
    env = dict(e.partition("=")[::2] for e in image_env or [])
    env.update({k: str(v) for k, v in (spec.get("environment") or {}).items()
                if v is not None})
    return [f"{k}={v}" for k, v in env.items()]


def _recreate(state: dict, svc: str) -> None:
    project = state.get("project", "sy")
    spec = state["services"][svc]
    tag = _tag(project, svc, spec)
    img = state.setdefault("images", {}).get(tag)
    if img is None:
        img = {"Id": f"sha256:pulled-{svc}", "Labels": {}}
        state["images"][tag] = img
    state.setdefault("containers", {})[svc] = {
        "ID": f"c-{svc}", "Image": img["Id"], "Env": _env_list(spec, img.get("Env")),
        "Status": "running",
        "Health": state.get("health_after", {}).get(svc, "healthy"),
    }


def main(argv: list[str]) -> int:
    log = os.environ.get("FAKE_DOCKER_LOG")
    if log:
        with open(log, "a") as fh:
            fh.write(" ".join(argv) + "\n")
    path = os.environ["FAKE_DOCKER_STATE"]
    with open(path, "r+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        state = json.load(fh)
        rc = handle(state, argv)
        fh.seek(0)
        fh.truncate()
        json.dump(state, fh, indent=1)
    return rc


def _c_config(state, rest):
    cfg = {"name": state.get("project", "sy"), "services": {}}
    for svc, spec in state["services"].items():
        out = {"environment": spec.get("environment") or {}}
        if spec.get("image"):
            out["image"] = spec["image"]
        if spec.get("build"):
            out["build"] = {"context": os.getcwd(), **spec["build"]}
        cfg["services"][svc] = out
    print("\n".join(state["services"]) if "--services" in rest else json.dumps(cfg))
    return 0


def _c_ps(state, rest):
    containers = state["containers"]
    if "-q" in rest:
        c = containers.get(rest[-1])
        if c and c.get("Status") == "running":
            print(c["ID"])
        return 0
    for svc, c in containers.items():
        print(json.dumps({"Service": c.get("Service", svc), "ID": c["ID"],
                          "State": c["Status"]}))
    return 0


def _c_port(state, rest):
    ans = state.get("ports", {}).get(rest[0]) if rest else None
    if not ans:
        return 1
    print(ans)
    return 0


def _c_build(state, rest):
    project = state.get("project", "sy")
    for svc in [x for x in rest if not x.startswith("-")]:
        tag = _tag(project, svc, state["services"][svc])
        n = state.get("builds", 0) + 1
        state["builds"] = n
        label = os.environ.get("SWITCHYARD_INPUTS_" + svc.upper().replace("-", "_"), "")
        state["images"][tag] = {"Id": f"sha256:build{n}-{svc}",
                                "Labels": {"switchyard.inputs": label} if label else {}}
    return 0


def _c_up(state, rest):
    project = state.get("project", "sy")
    flags = [x for x in rest if x.startswith("-")]
    targets = [x for x in rest if not x.startswith("-")] or list(state["services"])
    for svc in targets:
        c = state["containers"].get(svc)
        img = state["images"].get(_tag(project, svc, state["services"][svc]))
        if "--force-recreate" in flags or c is None:
            _recreate(state, svc)
        elif "--no-recreate" in flags:
            c["Status"] = "running"
        elif img and c.get("Image") != img["Id"]:
            _recreate(state, svc)
    return 0


def _c_stop(state, rest):
    for svc in rest:
        if svc in state["containers"]:
            state["containers"][svc]["Status"] = "exited"
    return 0


def _c_logs(state, rest):
    text = _pop(state, "logs", "")
    if text:
        print(text)
    return 0


def _redis(state, rest):
    if rest[:1] == ["redis-cli"]:
        rest = rest[1:]
    if rest[:1] == ["-n"]:
        rest = rest[2:]
    op = rest[0].upper() if rest else ""
    if op == "ZCARD":
        plan = rest[1].split("sy:inflight:", 1)[-1]
        print(_pop(state.setdefault("zcard", {}), plan, 0))
    elif op == "GET":
        print(state.get("redis_get", {}).get(rest[1], ""))
    elif op == "SCARD":
        print(0)
    elif op == "--SCAN":
        for plan in state.get("zcard", {}):
            print(f"sy:inflight:{plan}")
    elif op in ("SET", "DEL"):
        print("OK" if op == "SET" else 1)
    return 0


def _c_exec(state, rest):
    if rest and rest[0] == "-T":
        rest = rest[1:]
    svc, rest = rest[0], rest[1:]
    if svc == "redis":
        return _redis(state, rest)
    if svc == "gateway" and "switchyard.drain" in rest:
        print(f"migrated 0 session(s) off {rest[-1]}")
        return 0
    c = state["containers"].get(svc)
    if not c or c.get("Status") != "running":
        print(f"service {svc} is not running", file=sys.stderr)
        return 1
    if rest[:2] == ["python3", "-"]:
        sys.stdin.read()
        print(_pop(state.setdefault("health", {}), svc, "idle"))
        return 0
    if rest[:1] == ["sh"] and svc in state.get("preflight_fail", []):
        print("mkdir: Permission denied", file=sys.stderr)
        return 1
    return 0


COMPOSE = {"config": _c_config, "ps": _c_ps, "build": _c_build, "up": _c_up,
           "stop": _c_stop, "logs": _c_logs, "exec": _c_exec, "port": _c_port}


def _image_inspect(state, tag):
    img = state["images"].get(tag)
    if img is None:
        print(f"Error: No such image: {tag}", file=sys.stderr)
        print("[]")
        return 1
    print(json.dumps([{"Id": img["Id"], "Config": {"Labels": img.get("Labels") or {},
                                                   "Env": img.get("Env") or []}}]))
    return 0


def _inspect(state, args):
    fmt = None
    ids = []
    for x in args:
        if x.startswith("--format="):
            fmt = x.split("=", 1)[1]
        elif x == "--format":
            fmt = ""
        elif fmt == "":
            fmt = x
        else:
            ids.append(x)
    by_id = {c["ID"]: c for c in state["containers"].values()}
    if fmt is not None:
        c = by_id.get(ids[0]) if ids else None
        if c is None:
            return 1
        print(c.get("Health") or c.get("Status"))
        return 0
    docs = [{"Id": c["ID"], "Image": c["Image"], "Config": {"Env": c.get("Env") or []},
             "State": {"Status": c["Status"]}}
            for c in (by_id.get(cid) for cid in ids) if c]
    print(json.dumps(docs))
    return 0


def handle(state: dict, a: list[str]) -> int:
    for key in ("services", "containers", "images"):
        state.setdefault(key, {})
    if a[:1] == ["compose"] and len(a) > 1:
        fn = COMPOSE.get(a[1])
        return fn(state, a[2:]) if fn else 0
    if a[:2] == ["image", "inspect"]:
        return _image_inspect(state, a[2])
    if a[:1] == ["inspect"]:
        return _inspect(state, a[1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
