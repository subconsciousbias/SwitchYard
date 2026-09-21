# Working in this repo

## Never write to `.env`

`.env` holds real API keys and OAuth paths that the user typed in by hand. It is
gitignored, there is no backup, and **overwriting it destroys credentials that
cannot be recovered** — this has already happened once by way of
`cp .env.example .env`.

Rules, without exception:

- Never `cp`, `>`, `tee`, `sed -i`, or otherwise write to `.env`.
- When a new setting is needed, add it to **`.env.example` only**, then tell the
  user which key to fill in and where to get the value.
- To propagate new keys, run `scripts/sync-env.sh`. It appends missing keys,
  backs the file up first, and never touches an existing value.
- Read `.env` only when the user asks you to debug it, and never echo a secret's
  value into the transcript — print `KEY=<set>` instead.

## Config lives in `config/plans.yaml` — which is gitignored

`config/plans.yaml` is the operator's live portfolio and is NOT tracked.
`config/plans.example.yaml` is the tracked, genericised version. A change to the
schema, a new key or a new plan shape has to land in the EXAMPLE as well, or a
fresh clone gets a config that cannot express it.



Caps, cost, expiry, lane order, quota windows, credentials-by-env-var-name and
sidecar model aliases all come from that one file. `docker-compose.yml` must not
duplicate any of them — the sidecars read `plans.yaml` themselves. If you find
yourself setting the same number in two places, the config is the source of
truth and the other place is a bug.

## `switchyard/` is baked into three images, not mounted

Only `./config` is mounted. `switchyard/*.py` is COPYed into the gateway, the
portal and the sidecar images, so after editing it:

```bash
docker compose build gateway portal claude-max-sidecar xai-token-proxy
docker compose up -d
```

`docker compose restart` re-runs the OLD code and looks like the change did
nothing — or worse, half the stack picks it up and the other half does not, which
reads as an inconsistent bug. This has already cost three debugging detours: a
`hooks.py` change that "had no effect", a `models.py` schema addition that
crashed the portal on a config only it had, and a headroom calculation that was
right in the gateway and wrong on the board.

Two related traps in the same family:

- `.env` is read when a container is **created**. A value added afterwards needs
  `up -d --force-recreate <service>`, not `restart`.
- `config/litellm.generated.yaml` in the repo is a stale artifact. The gateway
  generates its own at `/tmp/litellm.generated.yaml` on startup; read that one.

## Don't touch credential stores

Same reasoning as `.env`: never run `docker login`/`docker logout`, never write to
the keychain, and never delete a stored token. Diagnose read-only, then give the
user the exact command to run themselves. Print `KEY=<set>` or a length, never a
secret's value.

Also: macOS has no `timeout(1)`. Use `gtimeout` if coreutils is installed, or
leave the command unbounded.

## Appending to a test file

`tests/*.py` end with an `if __name__ == "__main__":` runner that discovers tests
from `globals()`. Anything appended *after* that block is defined too late to be
collected, so it silently does not run — the suite still reports "N tests passed"
with your new test absent. Insert before the runner, and check the count went up.

## Tests run without services

`tests/*.py` are plain scripts, no pytest plugins, no Redis, no network:

```bash
python3 tests/test_routing.py tests/test_classify.py 2>/dev/null; \
for t in tests/test_*.py; do python3 "$t" >/dev/null || echo "FAIL $t"; done
python3 tests/render_preview.py
```

Keep it that way — anything needing a live provider belongs in `TESTING.md` as a
manual step instead.
