# Working in this repo

The same guardrails documented in [CLAUDE.md](CLAUDE.md) apply to every
agent runtime, not just Claude Code. OpenCode-driven runtimes pick them up
from `.opencode/opencode.json`. The rules they enforce:

- **Never write to `.env`.** Keys are typed by hand, gitignored, and cannot
  be recovered. Add new keys to `.env.example` only; run
  `scripts/sync-env.sh` to propagate. Read `.env` only when debugging.
  Guarded by `.claude/settings.json` denies (`Read(.env)`,
  `Edit(.env)`, `Write(.env)`) and `.claude/hooks/guard.sh` blocks
  `cp`/`mv`/`tee`/`sed -i`/redirects to `.env`.

- **This checkout is a worktree — commit here, never build or deploy from
  it.** The live stack runs from the main checkout; every worktree shares
  `docker-compose.yml`'s `${SWITCHYARD_PROJECT:?…}` name, so building /
  recreating here touches the live containers from branch code. Guarded
  by the top-of-file `exit 2` checks in `scripts/apply.sh` and
  `scripts/reload.sh`, by the `name: ${SWITCHYARD_PROJECT:?…}` fail-closed
  substitution in `docker-compose.yml`, and by `.claude/hooks/guard.sh`
  blocking `docker compose build`, `docker compose up`, `docker login`,
  and `docker logout` from inside a worktree.

- **Don't touch credential stores.** Never `docker login`/`docker logout`,
  never write to the keychain, never delete a stored token. Diagnose
  read-only, then give the exact command to the operator. Guarded by
  `.claude/settings.json` denies for `Bash(docker login:*)`,
  `Bash(docker logout:*)`, `Bash(security add-*)`, `Bash(security delete-*)`
  and by the hook's `security (add|delete)-` regex.

A block is a guard, not the prose — the prose in CLAUDE.md is for humans,
the guards are here so the prose is never the only thing in the way.
