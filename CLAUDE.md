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

## Config lives in `config/plans.yaml`

Caps, cost, expiry, lane order, quota windows, credentials-by-env-var-name and
sidecar model aliases all come from that one file. `docker-compose.yml` must not
duplicate any of them — the sidecars read `plans.yaml` themselves. If you find
yourself setting the same number in two places, the config is the source of
truth and the other place is a bug.

## Tests run without services

`tests/*.py` are plain scripts, no pytest plugins, no Redis, no network:

```bash
python3 tests/test_routing.py tests/test_classify.py 2>/dev/null; \
for t in tests/test_*.py; do python3 "$t" >/dev/null || echo "FAIL $t"; done
python3 tests/render_preview.py
```

Keep it that way — anything needing a live provider belongs in `TESTING.md` as a
manual step instead.
