#!/usr/bin/env bash
# One-off cleanup for the issue #132 existing-state mess:
# `~/.claude/projects` and `~/.codex/sessions` accumulate a per-run JSONL
# transcript every time the sidecars spawn a CLI. cli_bridge now reclaims
# each per-call project dir on the way out (so future runs are tidy), but
# everything written before that fix landed is still on disk.
#
# SAFETY: this script refuses to run unless BOTH target env vars are set
# explicitly. The bridges' defaults (``$HOME/.claude/projects``,
# ``$HOME/.codex/sessions``) are the host operator's own transcript
# stores -- not the sidecar mount -- so the documented no-env invocation
# is destructive on data the bridges never created. Reviewer finding,
# PR #322 cycle 2: refuse unless the operator points at the mounted path.
#
# Run this once from the operator's host to sweep that backlog, pointing
# at the sidecar mount (NOT the host's own ``~/.claude/projects``):
#
#     CLAUDE_PROJECTS_DIR=./secrets/claude/projects \
#     CODEX_SESSIONS_DIR=./secrets/codex/sessions \
#     bash scripts/clean-cli-transcripts.sh
#
# Each step prints every file or dir it would remove, then removes it.
# Set ``CLEAN_CLI_DRY_RUN=1`` to print without deleting (recommended on
# first run).
#
# Refuses to run if either env var was not set explicitly, or if BOTH
# target dirs are absent.
set -euo pipefail

# Refuse to fall back to the host defaults: those are the operator's
# own Claude Code / Codex transcript stores, NOT the sidecar mount, and
# removing them wipes data the bridges never created. ``${VAR+set}``
# expands to ``set`` when VAR is set (even to the empty string), empty
# otherwise -- the empty expansion is the test for "unset".
claude_set="${CLAUDE_PROJECTS_DIR+set}"
codex_set="${CODEX_SESSIONS_DIR+set}"
if [ -z "$claude_set" ] || [ -z "$codex_set" ]; then
  cat >&2 <<'EOF'
refusing to run: both CLAUDE_PROJECTS_DIR and CODEX_SESSIONS_DIR must
be set explicitly. The bridges' defaults ($HOME/.claude/projects and
$HOME/.codex/sessions) are the host operator's own transcript stores,
NOT the sidecar mount, and removing them would wipe host-owned data.

Run with both env vars pointing at the sidecar mount, e.g.:

    CLAUDE_PROJECTS_DIR=./secrets/claude/projects \
    CODEX_SESSIONS_DIR=./secrets/codex/sessions \
    bash scripts/clean-cli-transcripts.sh

Add CLEAN_CLI_DRY_RUN=1 to print what would be removed without deleting.
EOF
  exit 2
fi

# After the explicit-set guard, defaulting to empty strings lets the
# "refuse if both dirs absent" check below stay meaningful without
# re-checking the guard.
: "${CLAUDE_PROJECTS_DIR:=}"
: "${CODEX_SESSIONS_DIR:=}"

dry_run="${CLEAN_CLI_DRY_RUN:-0}"

claude_ok=0
codex_ok=0
if [ -d "$CLAUDE_PROJECTS_DIR" ]; then
  claude_ok=1
fi
if [ -d "$CODEX_SESSIONS_DIR" ]; then
  codex_ok=1
fi

if [ "$claude_ok" -eq 0 ] && [ "$codex_ok" -eq 0 ]; then
  echo "neither $CLAUDE_PROJECTS_DIR nor $CODEX_SESSIONS_DIR exists;" >&2
  echo "(set them to the sidecar mount if the sidecars put projects and" >&2
  echo "rollouts under a different name.)" >&2
  exit 1
fi

if [ "$claude_ok" -eq 1 ]; then
  echo "==> claude project dirs under $CLAUDE_PROJECTS_DIR"
  count=0
  while IFS= read -r -d '' entry; do
    count=$((count + 1))
    echo "  remove: $entry"
    if [ "$dry_run" = "0" ]; then
      rm -rf -- "$entry"
    fi
  done < <(find "$CLAUDE_PROJECTS_DIR" -mindepth 1 -maxdepth 1 -print0)
  echo "  $count entries"
fi

if [ "$codex_ok" -eq 1 ]; then
  echo "==> codex session rollouts under $CODEX_SESSIONS_DIR"
  count=0
  while IFS= read -r -d '' entry; do
    count=$((count + 1))
    echo "  remove: $entry"
    if [ "$dry_run" = "0" ]; then
      rm -rf -- "$entry"
    fi
  done < <(find "$CODEX_SESSIONS_DIR" -mindepth 1 -maxdepth 1 -print0)
  echo "  $count entries"
fi

if [ "$dry_run" = "1" ]; then
  echo
  echo "CLEAN_CLI_DRY_RUN=1 -- nothing was deleted"
fi