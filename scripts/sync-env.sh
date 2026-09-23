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
  echo "created $target from $example — fill in the blanks"
  exit 0
fi

# Back the file up before touching it at all. Cheap insurance — but OUTSIDE the
# worktree: these snapshots hold real keys, and four of them were once committed
# and pushed because .gitignore covered .env and not .env.backup.*. A file that
# cannot be added is better than one that must be remembered.
backup_dir="${SWITCHYARD_ENV_BACKUPS:-$HOME/.switchyard/env-backups}"
mkdir -p "$backup_dir"
backup="$backup_dir/env.$(date +%Y%m%d-%H%M%S)"
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
while IFS= read -r line; do
  case "$line" in
    ''|\#*) continue ;;
  esac
  key="${line%%=*}"
  if ! grep -q "^${key}=" "$target"; then
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
empty=$(grep -E '^[A-Z0-9_]+=$' "$target" | cut -d= -f1 || true)
if [ -n "$empty" ]; then
  echo
  echo "still empty (fill these in):"
  printf '  %s\n' $empty
fi
