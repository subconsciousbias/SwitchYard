#!/usr/bin/env bash
# PreToolUse hook for Claude Code, and a defense-in-depth check for any
# runtime that pipes tool calls through it. Wired in .claude/settings.json
# for Bash, Edit and Write tools. Reads ONE line of JSON on stdin shaped
# like Claude Code's PreToolUse payload:
#   {"tool_name":"Bash","tool_input":{"command":"..."}}
# Tolerates a $TOOL_NAME / $CMD env fallback so the hook can be invoked
# directly for tests and diagnostics.
#
# A deny prints a one-line reason to stderr and exits 2, which Claude Code
# treats as a tool-block (PreToolUse gates the call on exit code). Exit 0
# lets the call through. Keep the deny rules at the top so the audit trail
# is one place.

set -euo pipefail

# ---- deny rules ------------------------------------------------------------

# .env token: a standalone `.env`, not `.env.example`, `.env.backup.*`, or
# the trailing `.env` of an unrelated path like `/tmp/test.env`. Pre-context
# must be start-of-line or non-alnum/underscore (`test` fails because `t`
# is alnum); post-context must be end-of-line or non-alnum/underscore/dot
# (`.example` and `.backup.2024` fail because `.` is in the negation).
DOTENV_TOKEN='(^|[^[:alnum:]_])\.env($|[^[:alnum:]_.-])'

# .env write: an operator that puts bytes into a target path, anything that
# isn't a shell chain separator, then a tokenized `.env`. The operator list
# was widened past review (rm/dd/install/rsync also put bytes into a target
# path), and the `.env` boundary above stops `.env.example`,
# `.env.backup.*`, and unrelated paths ending in `.env` from matching.
DOTENV_OPS='(>|>>|tee|cp|mv|sed -i|rm|dd|install|rsync)'
DOTENV_WRITE_PATTERN="${DOTENV_OPS}[^|;&]*${DOTENV_TOKEN}"

# macOS keychain writes / deletes via `security(1)`.
SECURITY_PATTERN='security (add|delete)-'

# Worktree-only refuses — fine in the main checkout, refused from a worktree.
# docker-compose.yml pins ${SWITCHYARD_PROJECT}, so every worktree addresses
# the same compose project; building / up / login / logout here would touch
# the live stack from branch code.
#
# Anchored with ERE so we don't false-positive on `docker compose buildx`
# (matches `docker compose build` as a substring), `docker compose upgrade`
# (matches `docker compose up`), etc.
#
# Associative array so the operator-facing error prints the human-readable
# label (`docker compose build`) and not the regex literal
# (`(^|[[:space:]])docker[[:space:]]+compose[[:space:]]+build([[:space:]]|$)`).
declare -A WORKTREE_ONLY=(
  [docker login]='(^|[[:space:]])docker[[:space:]]+login([[:space:]]|$)'
  [docker logout]='(^|[[:space:]])docker[[:space:]]+logout([[:space:]]|$)'
  [docker compose build]='(^|[[:space:]])docker[[:space:]]+compose[[:space:]]+build([[:space:]]|$)'
  [docker compose up]='(^|[[:space:]])docker[[:space:]]+compose[[:space:]]+up([[:space:]]|$)'
)

# ---- extract command -------------------------------------------------------
# Stdin first (Claude Code), env vars as a fallback for direct invocation.
raw="$(head -c 65536 2>/dev/null || true)"

tool_name=""
command=""

if [ -n "$raw" ] && command -v python3 >/dev/null 2>&1 \
   && printf '%s' "$raw" | grep -q '"tool_input"'; then
  tool_name="$(printf '%s' "$raw" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    print(d.get("tool_name", ""))
except Exception:
    pass
')"
  command="$(printf '%s' "$raw" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    print(d.get("tool_input", {}).get("command", ""))
except Exception:
    pass
')"
fi

# Env fallback for direct invocation. Default to Bash so a hook with no
# tool name still goes through the regex checks below.
[ -z "$tool_name" ] && tool_name="${TOOL_NAME:-Bash}"
[ -z "$command" ] && command="${CMD:-}"

# Only Bash-shaped tools carry a command we can inspect. Read/Edit/Write
# are gated by permissions.deny in .claude/settings.json — Claude Code's
# own matcher runs before this hook. Anything non-Bash passes through.
case "$tool_name" in
  Bash|"") : ;;
  *) exit 0 ;;
esac

# ---- worktree detection ----------------------------------------------------
# Same idiom as scripts/apply.sh and scripts/reload.sh: --git-dir and
# --git-common-dir differ only inside a worktree. 2>/dev/null lets a
# non-repo invocation pass through (is_worktree stays 0).
is_worktree=0
if [ "$(git rev-parse --git-dir 2>/dev/null)" != "$(git rev-parse --git-common-dir 2>/dev/null)" ]; then
  is_worktree=1
fi

# ---- matches ---------------------------------------------------------------

# .env writes — deny everywhere. Reads (cat, grep, less) are not denied here:
# CLAUDE.md allows reading .env for debugging, and the value never appears in
# the transcript.
if printf '%s' "$command" | grep -E -q -- "$DOTENV_WRITE_PATTERN"; then
  echo "refusing: command would write to .env — keys there are typed by hand and cannot be recovered (see CLAUDE.md)" >&2
  exit 2
fi

# macOS keychain writes / deletes — deny everywhere.
if printf '%s' "$command" | grep -E -q -- "$SECURITY_PATTERN"; then
  echo "refusing: command would modify the macOS keychain via security(1) — see CLAUDE.md" >&2
  exit 2
fi

# Worktree-only refuses.
if [ "$is_worktree" -eq 1 ]; then
  for label in "${!WORKTREE_ONLY[@]}"; do
    pattern="${WORKTREE_ONLY[$label]}"
    if printf '%s' "$command" | grep -E -q -- "$pattern"; then
      echo "refusing: '$label' is refused from a worktree — exit the worktree and run from the main checkout (see CLAUDE.md)" >&2
      exit 2
    fi
  done
fi

exit 0
