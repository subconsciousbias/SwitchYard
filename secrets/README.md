# Sidecar credential stores

Each sidecar gets its **own** credential directory here, deliberately isolated
from your host CLIs. Log in once per sidecar:

```bash
docker compose exec claude-max-sidecar   claude login
docker compose exec codex-sidecar        codex login --device-auth
docker compose exec opencode-go-sidecar  opencode auth login --provider opencode-go
docker compose exec opencode-go2-sidecar opencode auth login --provider opencode-go
python3 -m switchyard.oauth login xai    # SuperGrok: SwitchYard's own grant,
                                         # stored as secrets/xai/oauth.json — no CLI
```

The xai-token-proxy container only sees `secrets/xai/` — its compose mount
is scoped to that subdirectory, so it can read and refresh its own grant
without ever being able to reach the Claude / Codex / OpenCode stores above.

`scripts/auth_audit.py` audits all of the above (and whatever sidecars
docker-compose.yml grows), and apply.sh runs the missing logins for you.

The logins persist here across restarts and rebuilds — the directory is a host
bind mount, so it is genuinely one-time. OpenCode keeps its login as
`auth.json` under its **data** directory (`secrets/opencode{,2}/data`), which
is also where `scripts/auth_audit.py` checks; the config directory holds only
settings.

**Device-code flows only.** A browser-callback login starts its listener inside
the container and points your host browser at `localhost:<port>`, which resolves
to your Mac rather than the container, so it never completes.
`codex login --device-auth` avoids that by giving you a code to type into the
website. Claude's login already works this way. OpenCode takes
`--provider <id>` to skip the picker, which matters over a `docker compose exec`
pipe.

## Why not just mount your host `~/.claude`?

Two reasons, both learned the hard way:

1. **It does not work for Claude on macOS.** Claude Code stores its OAuth token
   in the login Keychain, not in a file, so there is nothing for a Linux
   container to read. Mounting `~/.claude` gets you the settings and history but
   no credentials. The Linux equivalent is the user's keyring (GNOME Keyring /
   KWallet, accessed by the CLI via `secret-tool`); on Windows the token lives
   in Windows Credential Manager, encrypted with DPAPI. Either way, the
   container can't see it.

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
# macOS paths shown; on Linux substitute ~/.claude, ~/.codex,
# ~/.local/share/opencode, ~/.config/opencode; on Windows substitute
# %USERPROFILE%\.claude, %USERPROFILE%\.codex, %USERPROFILE%\.local\share\opencode,
# %USERPROFILE%\.config\opencode.
CLAUDE_CONFIG_DIR=/Users/you/.claude        # no credentials on macOS — settings only
CODEX_CONFIG_DIR=/Users/you/.codex
OPENCODE_DATA_DIR=/Users/you/.local/share/opencode
OPENCODE_CONFIG_DIR=/Users/you/.config/opencode
```

Accepting that the container may write to them while you are using them.

Nothing in this directory is committed — see `.gitignore`.
