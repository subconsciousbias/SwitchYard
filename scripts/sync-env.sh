#!/usr/bin/env bash
# Add keys that .env.example has and .env does not — and nothing else.
#
# This exists because `cp .env.example .env` destroys real credentials. This
# script only ever APPENDS missing keys; it never rewrites, reorders or clears a
# line that already exists, so running it against a filled-in .env is safe.
set -euo pipefail

cd "$(dirname "$0")/.."
example=".env.example"
target=".env"

[ -f "$example" ] || { echo "no $example here"; exit 1; }

# config/plans.yaml is your live portfolio and is gitignored, so a fresh clone
# has only the example. Nothing starts without it.
if [ ! -f config/plans.yaml ] && [ -f config/plans.example.yaml ]; then
  cp config/plans.example.yaml config/plans.yaml
  echo "created config/plans.yaml from the example — edit it to match your plans"
fi

if [ ! -f "$target" ]; then
  cp "$example" "$target"
  # Issue #118: the published `LITELLM_MASTER_KEY=sk-switchyard-change-me` is
  # a CRITICAL at startup, and the gateway's selfcheck refuses to boot on it.
  # Generate a random key and rewrite ONLY this one line in the freshly
  # created file. The rewrite is structural — it happens after cp, before any
  # operator edit, and only on a fresh create — so it cannot overwrite an
  # existing value: by definition no .env existed before this branch ran.
  generated_key=""
  if command -v openssl >/dev/null 2>&1; then
    generated_key="sk-$(openssl rand -hex 24)"
  elif command -v python3 >/dev/null 2>&1; then
    # POSIX fallback if openssl is missing (alpine without it, minimal
    # images, etc). python3 is in the gateway image but this script runs on
    # the operator's host, so prefer a host-side generator when available.
    # The `command -v` gate keeps a missing-python3 from turning this into
    # the literal string "sk-" (the prefix outside the subshell, with the
    # subshell swallowing its own non-zero exit via `2>/dev/null`); without
    # the gate we would write `LITELLM_MASTER_KEY=sk-` to .env, exit 0, and
    # hand the operator a credential that the gateway's selfcheck will
    # refuse on the next start anyway.
    generated_key="sk-$(python3 -c 'import secrets;print(secrets.token_hex(24))' 2>/dev/null)"
  fi
  if [ -n "$generated_key" ]; then
    # Rewrite ONLY the LITELLM_MASTER_KEY line in the freshly copied file.
    # We use a tempfile + mv rather than `sed -i` because BSD sed (macOS)
    # and GNU sed (Linux) take different arguments for in-place editing, and
    # portability matters more than a single fork here. The mv is atomic on
    # the same filesystem, so a partial write cannot leave an envfile with
    # an empty master-key slot.
    tmp_env="${target}.tmp.$$"
    sed "s|^LITELLM_MASTER_KEY=.*$|LITELLM_MASTER_KEY=${generated_key}|" \
      "$target" > "$tmp_env"
    mv "$tmp_env" "$target"
    echo "created $target from $example with a random LITELLM_MASTER_KEY — fill in the other blanks"
  else
    # No generator available: refuse to leave the published placeholder in
    # place, because the next `docker compose up -d` will refuse to start.
    # Better to error loudly here than to silently hand the operator a .env
    # they will copy back to source control wondering why nothing works.
    echo "ERROR: created $target but could not generate a random LITELLM_MASTER_KEY" >&2
    echo "       (no openssl or python3 on PATH)." >&2
    echo "       Edit $target and replace LITELLM_MASTER_KEY=sk-switchyard-change-me" >&2
    echo "       with a random sk- key of at least 32 characters before running" >&2
    echo "       docker compose up -d, or install openssl and re-run" >&2
    echo "       scripts/sync-env.sh." >&2
    exit 1
  fi
  exit 0
fi

# Back the file up before touching it at all. Cheap insurance — but OUTSIDE the
# worktree: these snapshots hold real keys, and four of them were once committed
# and pushed because .gitignore covered .env and not .env.backup.*. A file that
# cannot be added is better than one that must be remembered.
backup_dir="${SWITCHYARD_ENV_BACKUPS:-$HOME/.switchyard/env-backups}"
mkdir -p "$backup_dir"
backup="$backup_dir/env.$(date +%Y%m%d-%H%M%S).$$"
cp "$target" "$backup"
chmod 600 "$backup" 2>/dev/null || true

# Issue #217: if the target does not end in a newline, a `>>` append
# would merge into the last line, corrupting a real credential. Ensure
# a terminator before appending. (`tail -c1` of a file ending in \n
# prints just the newline, which od renders as `0a`; a file ending in
# any other byte prints that byte. `xxd` is not installed everywhere —
# `od` is POSIX.)
if [ -s "$target" ] && [ "$(tail -c1 "$target" | od -An -tx1 | tr -d ' \n')" != "0a" ]; then
  printf '\n' >> "$target"
fi

added=0
while IFS= read -r line || [ -n "$line" ]; do
  line=${line%$'\r'}
  case "$line" in
    ''|\#*) continue ;;
  esac
  case "$line" in
    export\ *) key="${line#export }" ; key="${key%%=*}" ;;
    *)          key="${line%%=*}" ;;
  esac
  if ! grep -Eq "^[[:space:]]*(export[[:space:]]+)?${key}[[:space:]]*=" "$target"; then
    printf '%s\n' "$line" >> "$target"
    echo "  + $key"
    added=$((added + 1))
  fi
done < "$example"

if [ "$added" -eq 0 ]; then
  rm -f "$backup"
  echo "$target already has every key in $example — nothing changed"
else
  echo "added $added key(s) to $target; previous version saved as $backup"
fi

# Report keys that exist but have no value, so nothing silently stays blank.
empty=$(tr -d '\r' < "$target" | grep -E '^[A-Z0-9_]+=$' | cut -d= -f1 || true)
if [ -n "$empty" ]; then
  echo
  echo "still empty (fill these in):"
  printf '  %s\n' $empty
fi
