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

These rules are enforced mechanically now: `.claude/settings.json` denies
`Read(.env)` / `Edit(.env)` / `Write(.env)`, and `.claude/hooks/guard.sh`
fires as a PreToolUse hook to block `cp`/`mv`/`tee`/`sed -i`/redirects to
`.env` before they run. A block is a guard, not the prose — the prose is
here for humans, the guards are here so the prose is never the only thing
in the way.

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

## This checkout may be a worktree — commit here, never build or deploy from it

Agent sessions may run in a `git worktree` of this repo. The way to tell is
mechanical, not heuristic:

```sh
git rev-parse --git-dir        # the worktree's private git dir
git rev-parse --git-common-dir  # the shared .git (main checkout's)
```

If the two differ, this is a worktree. In the main checkout they are identical,
and the build/deploy commands below are correct to run there. The rest of this
section is about the worktree case: edit, run the offline test suite, and
**commit on the worktree's branch** when the change is ready. The main checkout
is where the work *ships*: the rebuild and redeploy happen there after the
worktree branch is merged.

Two things must not happen on a worktree:

  1. **Never `docker compose build` / `up -d` / `up -d --force-recreate` here.**
     `docker-compose.yml` declares the project name with
     `${SWITCHYARD_PROJECT:?…}` (sourced from the operator's `.env` via
     `scripts/sync-env.sh`), so worktrees without `.env` fail closed at
     parse time with "run compose from the main checkout". Even when the
     project name does resolve, every worktree addresses the *same* compose
     project: building or recreating from a worktree does not spin up an
     isolated copy — it rebuilds and recreates the **live** containers from
     branch code, and mounts that worktree's `./config` into them. This
     was nearly done once from the issue-11 worktree and aborted by the
     user.

  2. **Never merge the worktree branch into main here.** The main checkout
     owns that handoff — running `git merge` from a worktree drags the
     worktree's branch into the main checkout's working tree, which then has
     to be rebuilt anyway. Just commit; let the merge happen at the main
     checkout (or via a PR).

`scripts/apply.sh` and `scripts/reload.sh` refuse to run from a worktree at
the top of the script (exit 2, before any Docker call), and
`.claude/hooks/guard.sh` blocks the same worktree-only commands as a
PreToolUse hook. A block is a guard, not the prose — the prose is here
for humans, the guards are here so the prose is never the only thing in
the way.

What "done on the worktree" looks like:

  * `git status` clean, `git log` shows the new commit on the worktree branch.
  * Offline test suite (`scripts/test.sh`) green.
  * The user has the new commit hash and the branch name; the rebuild/redeploy
    (`scripts/apply.sh --build` from the main checkout) is theirs to run.

Read-only diagnosis from a worktree is fine: `docker compose ps`, `docker logs`,
the portal board. None of those mutate the live stack.

## `switchyard/` is baked into four images, not mounted

Only `./config` is mounted. What ends up baked in each image depends on the
image — partial copies matter:

  * `Dockerfile.gateway:122` — `COPY switchyard /switchyard/switchyard`. The
    gateway image carries the **whole `switchyard/` package**.
  * `Dockerfile.portal:5` — `COPY switchyard /app/switchyard`. The portal image
    also carries the **whole package**.
  * `Dockerfile.token_proxy:17-18` — only `switchyard/oauth.py` and
    `switchyard/__init__.py`. The token-proxy image carries **only the OAuth
    refresh path** — nothing else in the package is reachable there.
  * `Dockerfile.sidecar:39-40` — only `switchyard/caller_env.py` and
    `switchyard/__init__.py`. The sidecar image carries **only `caller_env`**
    — the rest of the package (`models.py`, `picker.py`, `usage.py`, ...) is
    not in that image. (The sidecar's missing `models.py` is the
    `caller_environment` bug filed separately.)

So a whole-package edit invalidates the gateway and portal images, and an edit
to a partially-copied module has to be re-checked against what that image
actually copies. Editing `switchyard/oauth.py` invalidates gateway + portal +
token-proxy (the gateway and portal have the whole package; the token-proxy
copies just oauth). Editing `switchyard/caller_env.py` invalidates gateway +
portal + sidecar. Editing `switchyard/models.py` invalidates only gateway +
portal — the sidecar/token-proxy images do not see it.

The sidecar image is shared: one `switchyard-sidecar:latest` image serves
`claude-max-sidecar`, `codex-sidecar`, and `opencode-go-sidecar` /
`opencode-go2-sidecar`. `claude-max-sidecar` is the only service with a
`build:` line in `docker-compose.yml`; the rest reference the tag it produced
so compose does not build the same image four times.

After editing code that is baked into an image, rebuild and redeploy **from
the main checkout** (see the worktree section above — never from a worktree):

```bash
docker compose build gateway portal claude-max-sidecar xai-token-proxy
docker compose up -d
```

Only `claude-max-sidecar` declares `build:` for the sidecar image; rebuilding
the sidecar image rebuilds all four sidecar services. Rebuilding
`xai-token-proxy` is independent. Editing a partially-copied module does not
require rebuilding the gateway/portal images for that module alone, but those
images are also invalidated by *any* edit to a file inside the
`switchyard/` package because they COPY the whole directory.

`docker compose restart` re-runs the OLD code and looks like the change did
nothing — or worse, half the stack picks it up and the other half does not,
which reads as an inconsistent bug. This has already cost three debugging
detours: a `hooks.py` change that "had no effect", a `models.py` schema
addition that crashed the portal on a config only it had, and a headroom
calculation that was right in the gateway and wrong on the board.

Two related traps in the same family:

- `.env` is read when a container is **created**. A value added afterwards needs
  `up -d --force-recreate <service>`, not `restart`.
- `config/litellm.generated.yaml` is **gitignored** (see `.gitignore:14`); the
  gateway regenerates it as `/tmp/litellm.generated.yaml` on every container
  start (`docker/gateway-entrypoint.sh:9`), so reading it on the host reads a
  stale file. Read it inside the gateway container instead. To apply an edit to
  `config/plans.yaml` against the running stack, use `scripts/reload.sh`: it
  validates first, then asks the gateway whether its router can follow the edit;
  on a policy-only change (caps, quotas, lane order, settings) the gateway
  hot-swaps its registry within ~5s, and only a router-shaped change forces
  `docker compose restart gateway`. The script then refreshes the portal board
  via `POST /admin/reload` (portal-board-only effect — the sidecars re-read
  the file themselves within 30 seconds).

## Don't touch credential stores

Same reasoning as `.env`: never run `docker login`/`docker logout`, never write
to the keychain, and never delete a stored token. Diagnose read-only, then give
the user the exact command to run themselves. Print `KEY=<set>` or a length,
never a secret's value.

These rules are enforced mechanically now: `.claude/settings.json` denies
`Bash(docker login:*)` / `Bash(docker logout:*)` /
`Bash(security add-*)` / `Bash(security delete-*)`, and
`.claude/hooks/guard.sh` fires as a PreToolUse hook to block the same
shapes (including `security(1)` keychain writes) before they run. A
block is a guard, not the prose — the prose is here for humans, the
guards are here so the prose is never the only thing in the way.

Also: macOS has no `timeout(1)`. Use `gtimeout` if coreutils is installed, or
leave the command unbounded.

## Per-directory CLAUDE.md

Repo-wide safety rules live here. Test conventions, sidecar/bridge details, and
portal parity rules have moved to per-directory files:

  * `sidecars/CLAUDE.md` — image contents, sidecar env/tunables, the
    bridge-siblings rule.
  * `tests/CLAUDE.md` — the plain-script runner, conftest's socket guard and
    its real scope, `tests/fake_redis.py`, how to add a new test file.
  * `switchyard/portal/CLAUDE.md` — board/gateway parity rule and Jinja2
    template conventions.

Where a mechanical guard already exists, those files point to it instead of
restating prose (notably `tests/conftest.py`). Where none exists yet
(bridge-sibling diff, example-parity), the rule lives in prose as the interim
form.
