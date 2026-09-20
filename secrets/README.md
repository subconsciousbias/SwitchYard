# Sidecar credential stores

Each sidecar gets its **own** credential directory here, deliberately isolated
from your host CLIs. Log in once per sidecar:

```bash
docker compose exec claude-max-sidecar   claude login
docker compose exec codex-sidecar        codex login
docker compose exec grok-sidecar         opencode auth login   # choose xAI
docker compose exec opencode-go-sidecar  opencode auth login   # choose OpenCode Zen
```

The logins persist here across restarts and rebuilds.

## Why not just mount your host `~/.claude`?

Two reasons, both learned the hard way:

1. **It does not work for Claude on macOS.** Claude Code stores its OAuth token
   in the login Keychain, not in a file, so there is nothing for a Linux
   container to read. Mounting `~/.claude` gets you the settings and history but
   no credentials.

2. **It is not safe.** The container needs write access — OAuth refresh rotates
   the token and has to persist it — and that means a containerised CLI writing
   into the very config directory your interactive CLI is using. It can rewrite
   `.claude.json`, settings and session state underneath a running session.

Isolated stores also mean a sidecar's token refresh can never invalidate the
session you are using interactively: two logins on one account are independent,
whereas one refresh-token chain copied into two places can rotate out from under
the other.

## Using host credentials anyway

If you would rather share (it does work for Codex and OpenCode, which are
file-based), point the relevant variable in `.env` at your host directory:

```
CLAUDE_CONFIG_DIR=/Users/you/.claude        # no credentials on macOS — settings only
CODEX_CONFIG_DIR=/Users/you/.codex
OPENCODE_DATA_DIR=/Users/you/.local/share/opencode
OPENCODE_CONFIG_DIR=/Users/you/.config/opencode
```

Accepting that the container may write to them while you are using them.

Nothing in this directory is committed — see `.gitignore`.
