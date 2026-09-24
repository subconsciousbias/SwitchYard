# Working in `sidecars/`

This directory holds the two sidecar images that run alongside the gateway:
the sidecar image (shared by four subscription services) and the token-proxy
image (one service). Only `./config` is mounted into them at runtime;
everything in `sidecars/` and the small slivers of `switchyard/` that get
baked in are part of the image itself, so a code edit there is followed by
`scripts/apply.sh` from the main checkout, which detects the changed inputs
and rebuilds the sidecar image (see the root `CLAUDE.md`).

## Image contents

What ends up baked in each image is a deliberate partial copy — not all four
images carry the full `switchyard/` package:

  * `Dockerfile.sidecar` — the **sidecar image** (`switchyard-sidecar:latest`),
    shared by `claude-max-sidecar`, `codex-sidecar`, `opencode-go-sidecar`, and
    `opencode-go2-sidecar`. It carries:
      - `sidecars/cli_bridge/server.py` (the cli_bridge bridge)
      - `sidecars/mcp_bridge/` (the mcp_bridge bridge, whole directory)
      - `sidecars/cli_bridge/harness/` (the harness configs both bridges need)
      - `switchyard/__init__.py`, `switchyard/caller_env.py`, and
        `switchyard/models.py` — `models.py` ships because the bridges
        build `CallerEnvironmentSettings` from
        `settings.caller_environment` in plans.yaml (issue #44 +
        operator config); dropping it silently turns that config into
        `None` on every request. None of `picker.py`, `usage.py`,
        `hooks.py`, `oauth.py`, etc.
    `claude-max-sidecar` is the only service with a `build:` line in
    `docker-compose.yml`; the others reference the tag it produced.

  * `Dockerfile.token_proxy` — the **token-proxy image**
    (`switchyard-token-proxy:latest`), used by `xai-token-proxy`. It carries:
      - `sidecars/token_proxy/server.py`
      - `switchyard/__init__.py` and `switchyard/oauth.py` only — the OAuth
        refresh path is what the proxy needs; nothing else in the package is
        in this image.

The full `switchyard/` package is baked into `gateway` and `portal` by their
Dockerfiles (`Dockerfile.gateway:122` and `Dockerfile.portal:5`); those are the
two images that *can* see `models.py`, `picker.py`, etc. Sidecar/token-proxy
edits to a module not in their partial copy are no-ops at best — verify what
each image actually copies before relying on a change taking effect there.

## Sidecar env / tunables

The sidecar image sets up the runtime environment in `Dockerfile.sidecar`:

  - `HOME=/home/node` plus `XDG_DATA_HOME`, `XDG_CONFIG_HOME`, `XDG_CACHE_HOME`,
    `XDG_STATE_HOME` all pointing at `/home/node/...`. (Pre-#223 the
    container ran as root and `$HOME` would otherwise default to `/root`;
    Claude and Codex are pinned by `CLAUDE_CONFIG_DIR`/`CODEX_HOME` and were
    unaffected, but OpenCode resolves XDG paths from `$HOME` and so looked
    in `/root`, finding zero credentials while a perfectly good login sat
    mounted a directory away. Pointing HOME and the XDG variables at the
    mounts is what fixed that. Post-#223 the sidecars run as `user: node`,
    so the four XDG dirs (`/home/node/.local/{state,share}`,
    `/home/node/.config`, `/home/node/.cache`) are pre-created in the
    image and their three parents (`/home/node/.local`,
    `/home/node/.config`, `/home/node/.cache`) are chowned to `node:node`
    — see `Dockerfile.sidecar`, and `SUBPROCESS_ENV_KEYS` in
    `cli_bridge/server.py` for the inner-CLI
    env allowlist that carries these pins to the spawned subprocess.)
  - `PROVIDER=claude`, `SWITCHYARD_PLANS=/app/config/plans.yaml`,
    `PYTHONPATH=/app`, `BRIDGE` (defaults to `cli`, set per service in compose),
    `SIDECAR_PORT` (the port this process listens on).

Knobs read by the running bridges (all read once at import time):

  - `CLI_EXTRA_ARGS` — extra argv to splice in for CLI-version-specific flags
    (`--full-auto`, etc.). Read in `cli_bridge/server.py`; consumed by both
    bridges because mcp_bridge reuses cli_bridge's argv builder.
  - `SIDECAR_TIMEOUT`, `SYSTEM_MODE`, `BARE`, `MCP_STDIN_PROMPT_LIMIT` —
    cli_bridge knobs.
  - `MCP_SESSION_TTL_SECONDS`, `MCP_REAP_INTERVAL_SECONDS`,
    `MCP_DISCONNECT_POLL_SECONDS`, `MCP_PARKED_GRACE_SECONDS`,
    `MCP_RESUME_WAIT_SECONDS`, `MCP_REBUILD_LOST`, `MCP_PREEMPTED_MEMORY`,
    `MCP_RECLAIM_POLL_SECONDS`, `MCP_PROCESS_TIMEOUT_SECONDS`,
    `MCP_BATCH_WINDOW_SECONDS`, `MCP_REMEMBERED_TOOLS_LIMIT`,
    `MCP_WORKDIR_ROOT`,
    `LOG_TEXT_LOST` — mcp_bridge knobs.
  - `MCP_PDF_PAGE_LIMIT`, `MCP_PDF_RENDER_DPI` — the PDF render
    (`cli_bridge.render_pdf`, poppler): pages rendered and their DPI, for a
    PDF in a tool result (claude/codex) and a PDF in a codex/opencode
    prompt. Defined in cli_bridge, reused by mcp_bridge.

The tool path runs each inner CLI in a mirror of the caller's cwd
(`mcp_bridge.acquire_mirror`, issue #264). `MCP_MIRROR_LIMIT` caps how many
distinct mirrors exist at once (default 64; past it a session keeps its
workdir) and `MCP_MIRROR_MANIFEST` is where created mirrors are recorded so
startup can sweep what a crash left behind.

Each session's workdir lands at `/relay/<session_id>` (issue #294), where
`/relay` is created by `Dockerfile.sidecar` as sticky + world-writable like
the mirror roots — the unprivileged `node` user can `mkdir` under it, and
`cleanup_workdir`'s `rmtree` removes exactly the session dir, never the
root. `MCP_WORKDIR_ROOT` overrides the default (`/relay`) when an image or
env lacks the dir; `new_workdir` falls back to the original
`tempfile.mkdtemp(prefix="mcpb-<id8>-")` rather than failing. None of this
touches the `mirror_target`/`MIRROR_ROOTS`/`MIRROR_DENY` path; the relay's
own paths (`/app`, `/relay/<session_id>`, `/tmp/sy-cli-*`) are named in
`render_first_turn_reminder`'s "trust the relay env" warning so the model
does not mistake them for the caller's filesystem.

`SWITCHYARD_PLAN` (or its legacy alias `SWITCHYARD_SUBSCRIPTION`) names the
plan this process fronts. One process per subscription: sharing a process
across plans would conflate their connection gates and quota.

The token-proxy image is smaller still:

  - `SWITCHYARD_PROVIDER` — `xai` or `openai`, one process per provider.
  - `SWITCHYARD_AUTH_STORE=/app/secrets/oauth.json` — read-write mount of
    `./secrets/xai`, so the proxy can refresh the access token in place. The
    mount is scoped to just this provider's grant directory; the other
    `./secrets/<provider>` stores (claude, codex, opencode, ...) are not
    reachable from this container.

## The bridge-siblings rule

`sidecars/cli_bridge/server.py` and `sidecars/mcp_bridge/server.py` are
siblings: mcp_bridge loads cli_bridge by file path
(`sidecars/mcp_bridge/server.py:81-85`) and reuses its argv building,
error-classification, profile, gate, and health logic rather than
duplicating them. A fix or feature in one bridge must be checked against the
other — the lock-step is not enforced mechanically, prose is the interim rule.

Recurring examples from git history that have to stay in step:

  - `is_error` mapping (`cli_bridge/server.py` classifies an errored CLI event;
    mcp_bridge reuses the same flag on the MCP side — `mcp_bridge/server.py`
    reads `msg.get("is_error")` on the OpenAI-shaped content it forwards).
  - `CLI_EXTRA_ARGS` — defined in cli_bridge (`os.environ.get("CLI_EXTRA_ARGS")`),
    read by mcp_bridge because mcp_bridge builds argv through cli_bridge too.
  - `max_tokens` / `enforce_max_tokens` profile fields — defined in
    cli_bridge's PROFILE (`enforce_max_tokens: bool`, plus
    `enforce_max_tokens_reason` on lanes that don't honor it), read by
    mcp_bridge's `/health` via `cli_bridge.PROFILE[...]`. Adding a new lane or
    a new refusal reason here has to land in both health endpoints together.

`_warm` (the cli_bridge startup-warm `asyncio.Event`) lives on cli_bridge only:
cli_bridge's `/health` reports it (`sidecars/cli_bridge/server.py:1636`), and
mcp_bridge's `/health` does not consult it. The cli_bridge module that
mcp_bridge loads via `importlib.util` (`sidecars/mcp_bridge/server.py:81-85`)
is a process-local copy; `_warm` is only set when `cli_bridge.invoke` runs
(`sidecars/cli_bridge/server.py:629-640`), which mcp_bridge never calls, so
reading `cli_bridge._warm.is_set()` from mcp_bridge's process always reads
`False`. The two services also run in different containers (`docker-compose.yml`
`BRIDGE=cli` vs `BRIDGE=mcp`), so there is no cross-process event to share.

If you find yourself reaching for a copy of any of these in mcp_bridge, that
is the signal to put it on cli_bridge and have mcp_bridge import it.
