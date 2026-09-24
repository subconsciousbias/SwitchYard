#!/usr/bin/env python3
"""Decide which baked images need a rebuild and which containers a recreate.

Used by scripts/apply.sh; runnable on its own:

    python3 scripts/image_plan.py plan [--force] [--no-build] [--tsv FILE]
    python3 scripts/image_plan.py build SERVICE...
    python3 scripts/image_plan.py inputs SERVICE        # debug: list the manifest

How the rebuild decision works -- content, not mtimes, not filenames:

  * Every service with a `build:` section in docker-compose.yml owns one image.
    Its build INPUTS are its Dockerfile plus every source path named on the
    Dockerfile's COPY/ADD lines (read from the Dockerfile itself, so a new COPY
    line is picked up without touching this script).
  * The inputs are hashed per file into a MANIFEST ("v1,path:sha12,..."). A
    directory COPY contributes every file under it (__pycache__/*.pyc and
    .DS_Store excluded -- interpreter/Finder by-products, not code).
  * `build` passes the manifest to `docker compose build` through
    SWITCHYARD_INPUTS_<SERVICE>, which docker-compose.yml interpolates into the
    image LABEL `switchyard.inputs`. The label therefore records exactly what
    the image was built from.
  * `plan` compares the live manifest against the label on the image tag. Equal
    means up to date; different names the files that changed; a missing label
    (an image built before this existed, or by a bare `docker compose build`)
    counts as stale, so the first run after this lands rebuilds once.

Which containers need a recreate -- exact, from Docker's own state:

  * every consumer of an image that is about to be rebuilt;
  * a consumer whose container runs an older image ID than the tag (built but
    never recreated);
  * a consumer whose container environment differs from what compose would
    create now (`docker compose config` resolves .env interpolation) -- that is
    how a .env edit is detected. Both directions count: a key compose sets
    that the container lacks or holds differently, and a key the container
    still carries that compose no longer sets (the image's own ENV excepted).
    Values are compared in memory and never printed; only KEY names are
    reported;
  * a consumer with no container at all (new service, or removed).

A service with several containers (scaled, or a stale one left by a
half-finished recreate) has every container checked, not just the first.

Only `docker compose config`, `docker image inspect`, `docker compose ps` and
`docker inspect` are called by `plan`; it never mutates anything.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shlex
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL = "switchyard.inputs"
SKIP_DIRS = {"__pycache__", ".git"}
SKIP_SUFFIXES = (".pyc", ".pyo")
SKIP_NAMES = {".DS_Store"}


# ------------------------------------------------------------------ inputs


def env_var_for(service: str) -> str:
    """SWITCHYARD_INPUTS_<SERVICE>: the compose interpolation variable that
    carries a service's manifest into its image label."""
    return "SWITCHYARD_INPUTS_" + service.upper().replace("-", "_")


def _logical_lines(text: str):
    buf = ""
    for raw in text.splitlines():
        line = raw.rstrip()
        if not buf and line.lstrip().startswith("#"):
            continue
        if line.endswith("\\"):
            buf += line[:-1] + " "
            continue
        buf += line
        if buf.strip():
            yield buf.strip()
        buf = ""
    if buf.strip():
        yield buf.strip()


def copy_sources(dockerfile_text: str) -> list[str]:
    """Source paths of every COPY/ADD that reads the build context.

    Skips `--from=` (copies from another stage/image, not the context) and
    strips the other `--flag` options (`--chown`, `--chmod`, `--link`, ...).
    Handles both the shell form and the JSON-array form.
    """
    out: list[str] = []
    for line in _logical_lines(dockerfile_text):
        parts = line.split(None, 1)
        if len(parts) < 2 or parts[0].upper() not in ("COPY", "ADD"):
            continue
        rest = parts[1].strip()
        args: list[str] = []
        flags: list[str] = []
        # flags come first in either form
        while rest.startswith("--"):
            flag, _, rest = rest.partition(" ")
            flags.append(flag)
            rest = rest.strip()
        if any(f.startswith("--from") for f in flags):
            continue
        if rest.startswith("["):
            try:
                args = [str(a) for a in json.loads(rest)]
            except ValueError:
                args = shlex.split(rest)
        else:
            args = shlex.split(rest)
        out.extend(args[:-1])   # the last argument is the destination
    return out


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    digest = h.hexdigest()[:12]
    if os.stat(path).st_mode & 0o111:
        digest += "+x"   # COPY preserves the mode bit; a chmod is a real change
    return digest


def _skip(name: str) -> bool:
    return name in SKIP_NAMES or name.endswith(SKIP_SUFFIXES)


def _expand(context: str, src: str) -> list[str]:
    """Context-relative files one COPY source contributes."""
    src = src.lstrip("/")
    matches = sorted(glob.glob(os.path.join(context, src))) if any(
        c in src for c in "*?[") else [os.path.join(context, src)]
    files: list[str] = []
    for m in matches:
        if os.path.isdir(m):
            for dirpath, dirnames, filenames in os.walk(m, followlinks=True):
                dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
                for f in sorted(filenames):
                    if not _skip(f):
                        files.append(os.path.join(dirpath, f))
        elif os.path.isfile(m):
            files.append(m)
        else:
            # A source that does not exist will fail the build itself; record
            # it so the manifest (and so the decision) still changes.
            files.append(m)
    return files


def manifest(context: str, dockerfile: str) -> dict[str, str]:
    """{context-relative path: sha12[+x]} for the Dockerfile and its inputs."""
    df_path = dockerfile if os.path.isabs(dockerfile) else os.path.join(context, dockerfile)
    entries: dict[str, str] = {}
    paths = [df_path]
    if os.path.isfile(df_path):
        with open(df_path, encoding="utf-8") as fh:
            for src in copy_sources(fh.read()):
                paths.extend(_expand(context, src))
    for p in paths:
        rel = os.path.relpath(p, context)
        entries[rel] = _hash_file(p) if os.path.isfile(p) else "missing"
    return entries


def encode(entries: dict[str, str]) -> str:
    return "v1," + ",".join(f"{k}:{v}" for k, v in sorted(entries.items()))


def decode(label: str) -> dict[str, str] | None:
    if not label or not label.startswith("v1,"):
        return None
    out: dict[str, str] = {}
    for item in label[3:].split(","):
        k, sep, v = item.rpartition(":")
        if sep:
            out[k] = v
    return out


def diff(old: dict[str, str], new: dict[str, str]) -> list[str]:
    changed = []
    for k in sorted(set(old) | set(new)):
        if k not in old:
            changed.append(f"+{k}")
        elif k not in new:
            changed.append(f"-{k}")
        elif old[k] != new[k]:
            changed.append(k)
    return changed


def summarise(paths: list[str], limit: int = 6) -> str:
    shown = ", ".join(paths[:limit])
    if len(paths) > limit:
        shown += f", +{len(paths) - limit} more"
    return shown


# ------------------------------------------------------------------ docker


def _docker(args: list[str], cwd: str) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], cwd=cwd, capture_output=True, text=True)


def compose_config(root: str) -> dict:
    p = _docker(["compose", "config", "--format", "json"], root)
    if p.returncode != 0:
        # stderr can quote the offending line of .env; keep only the tail
        msg = (p.stderr or "").strip().splitlines()[-1:] or ["(no output)"]
        raise SystemExit(f"image_plan: `docker compose config` failed: {msg[0]}")
    return json.loads(p.stdout)


def build_services(cfg: dict) -> dict[str, dict]:
    """{build service: {"image": tag, "context": dir, "dockerfile": path}}."""
    project = cfg.get("name") or os.path.basename(ROOT)
    out = {}
    for svc, spec in sorted((cfg.get("services") or {}).items()):
        build = spec.get("build")
        if not build:
            continue
        if isinstance(build, str):
            build = {"context": build}
        out[svc] = {
            "image": spec.get("image") or f"{project}-{svc}",
            "context": build.get("context") or ".",
            "dockerfile": build.get("dockerfile") or "Dockerfile",
        }
    return out


def _with_tag(image: str) -> str:
    last = image.rsplit("/", 1)[-1]
    return image if (":" in last or "@" in image) else image + ":latest"


def image_info(tag: str, root: str) -> dict | None:
    p = _docker(["image", "inspect", tag], root)
    if p.returncode != 0:
        return None
    try:
        data = json.loads(p.stdout)
    except ValueError:
        return None
    return data[0] if data else None


def containers(root: str) -> dict[str, list[dict]]:
    """{service: docker-inspect docs of EVERY container it has}, running ones
    first. A service can have more than one (a scaled service, or a stale
    container left by a half-finished recreate); the plan checks them all, so
    a stale one cannot hide behind a current one."""
    p = _docker(["compose", "ps", "-a", "--format", "json"], root)
    if p.returncode != 0 or not p.stdout.strip():
        return {}
    text = p.stdout.strip()
    rows = json.loads(text) if text.startswith("[") else [
        json.loads(ln) for ln in text.splitlines() if ln.strip()]
    ids: dict[str, list[str]] = {}
    for r in rows:
        svc, cid = r.get("Service"), r.get("ID")
        if svc and cid:
            ids.setdefault(svc, []).append(cid)
    if not ids:
        return {}
    q = _docker(["inspect", *(c for cs in ids.values() for c in cs)], root)
    docs = json.loads(q.stdout) if q.stdout.strip() else []
    out: dict[str, list[dict]] = {}
    for svc, cids in ids.items():
        found = [d for d in docs for cid in cids if d.get("Id", "").startswith(cid)]
        if found:
            found.sort(key=lambda d: ((d.get("State") or {}).get("Status") != "running"))
            out[svc] = found
    return out


def _env_map(items: list[str] | None) -> dict[str, str]:
    out = {}
    for item in items or []:
        k, _, v = item.partition("=")
        out[k] = v
    return out


def env_drift(expected: dict, container_env: list[str],
              image_env: list[str] | None = None) -> tuple[list[str], list[str]]:
    """(changed, removed) KEY names -- never values.

    changed: compose sets KEY and the container lacks it or has another value.
    removed: the container has KEY, compose no longer sets it, and it is not
    simply the image's own ENV (PATH, PYTHON_VERSION, ...) -- a variable
    dropped from compose/.env that the old container still carries."""
    have = _env_map(container_env)
    base = _env_map(image_env)
    expected = expected or {}
    changed = [k for k, v in sorted(expected.items())
               if v is not None and have.get(k) != str(v)]
    removed = []
    for k, v in sorted(have.items()):
        if k in expected:
            continue
        if k in base and base[k] == v:
            continue
        if k == "PATH" and "PATH" not in base:   # the daemon's default PATH
            continue
        removed.append(k)
    return changed, removed


# ------------------------------------------------------------------ plan


def make_plan(root: str, force: bool = False, no_build: bool = False) -> dict:
    cfg = compose_config(root)
    services = cfg.get("services") or {}
    builds = build_services(cfg)
    ctrs = containers(root)

    images = []      # [(svc, action, reason)]
    tag_ids = {}     # tag -> image id AFTER this run (None when being rebuilt)
    tag_env = {}     # tag -> the image's own ENV (a container inherits it)
    rebuilt_tags = set()
    for svc, b in builds.items():
        ctx = b["context"] if os.path.isabs(b["context"]) else os.path.join(root, b["context"])
        want = manifest(ctx, b["dockerfile"])
        tag = _with_tag(b["image"])
        info = image_info(tag, root)
        have = decode(((info or {}).get("Config") or {}).get("Labels", {}).get(LABEL, "")
                      if info else "")
        if info is None:
            reason = "no image yet"
        elif have is None:
            reason = f"image has no {LABEL} label (built before auto-detect, or by a bare `docker compose build`)"
        else:
            changed = diff(have, want)
            reason = ("changed: " + summarise(changed)) if changed else ""
        if force:
            reason = "forced (--build)" + (f"; {reason}" if reason else "")
        tag_ids[tag] = (info or {}).get("Id")
        tag_env[tag] = ((info or {}).get("Config") or {}).get("Env") or []
        if reason and not no_build:
            images.append((svc, "build", reason))
            rebuilt_tags.add(tag)
        elif reason:
            images.append((svc, "stale", reason + " -- skipped (--no-build)"))
        else:
            images.append((svc, "ok", "up to date"))

    # Consumers: every service whose image is one of the built tags.
    owner_of_tag = {_with_tag(b["image"]): svc for svc, b in builds.items()}
    svc_rows = []    # (svc, owner, plan, state)
    recreate = []    # (svc, reason)
    for svc, spec in sorted(services.items()):
        tag = _with_tag(spec.get("image") or builds.get(svc, {}).get("image", ""))
        if svc in builds:
            tag = _with_tag(builds[svc]["image"])
        owner = owner_of_tag.get(tag)
        if not owner:
            continue
        env = spec.get("environment") or {}
        plan = env.get("SWITCHYARD_PLAN") or "-"
        docs = ctrs.get(svc) or []
        state = ((docs[0] if docs else {}).get("State") or {}).get("Status") or "absent"
        svc_rows.append((svc, owner, plan, state))
        reasons = []
        if tag in rebuilt_tags:
            reasons.append(f"image {owner} rebuilt")
        if not docs:
            reasons.append("no container")
        for doc in docs:
            if tag not in rebuilt_tags and tag_ids.get(tag) and doc.get("Image") != tag_ids[tag]:
                reasons.append("container runs an older image than the tag")
            changed, removed = env_drift(env, (doc.get("Config") or {}).get("Env") or [],
                                         tag_env.get(tag))
            if changed:
                reasons.append("environment changed: " + summarise(changed))
            if removed:
                reasons.append("environment no longer set: " + summarise(removed))
        reasons = list(dict.fromkeys(reasons))
        if reasons:
            recreate.append((svc, "; ".join(reasons)))
    return {"images": images, "services": svc_rows, "recreate": recreate}


def print_human(plan: dict) -> None:
    print("    images (content hash of Dockerfile + COPY inputs vs the image's"
          f" {LABEL} label):")
    for svc, action, reason in plan["images"]:
        verb = {"build": "REBUILD", "stale": "stale", "ok": "ok"}[action]
        print(f"      {svc:<22} {verb:<8} {reason}")
    if plan["recreate"]:
        print("    containers to recreate:")
        for svc, reason in plan["recreate"]:
            print(f"      {svc:<22} {reason}")
    else:
        print("    containers: all current (image and environment)")


def write_tsv(plan: dict, path: str) -> None:
    with open(path, "w") as fh:
        for svc, action, reason in plan["images"]:
            if action == "build":
                fh.write(f"build\t{svc}\t{reason}\n")
        for svc, owner, p, state in plan["services"]:
            fh.write(f"service\t{svc}\t{owner}\t{p}\t{state}\n")
        for svc, reason in plan["recreate"]:
            fh.write(f"recreate\t{svc}\t{reason}\n")


# ------------------------------------------------------------------ build


def build(root: str, targets: list[str]) -> int:
    cfg = compose_config(root)
    builds = build_services(cfg)
    unknown = [t for t in targets if t not in builds]
    if unknown:
        print(f"image_plan: not a build service: {' '.join(unknown)}", file=sys.stderr)
        return 2
    env = os.environ.copy()
    # Every build service's label var is set, not just the targets': compose
    # interpolates the whole file, and a stale value would never be read.
    for svc, b in builds.items():
        ctx = b["context"] if os.path.isabs(b["context"]) else os.path.join(root, b["context"])
        env[env_var_for(svc)] = encode(manifest(ctx, b["dockerfile"]))
    return subprocess.call(["docker", "compose", "build", *targets], cwd=root, env=env)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=ROOT, help="compose project directory")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--force", action="store_true", help="rebuild every image")
    p.add_argument("--no-build", action="store_true", help="never rebuild")
    p.add_argument("--tsv", help="also write a machine-readable plan here")
    b = sub.add_parser("build")
    b.add_argument("services", nargs="+")
    i = sub.add_parser("inputs")
    i.add_argument("service")
    args = ap.parse_args(argv)
    root = os.path.abspath(args.root)

    if args.cmd == "plan":
        plan = make_plan(root, force=args.force, no_build=args.no_build)
        print_human(plan)
        if args.tsv:
            write_tsv(plan, args.tsv)
        return 0
    if args.cmd == "build":
        return build(root, args.services)
    cfg = compose_config(root)
    b = build_services(cfg).get(args.service)
    if not b:
        print(f"image_plan: not a build service: {args.service}", file=sys.stderr)
        return 2
    ctx = b["context"] if os.path.isabs(b["context"]) else os.path.join(root, b["context"])
    for k, v in sorted(manifest(ctx, b["dockerfile"]).items()):
        print(f"{v:<16} {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
