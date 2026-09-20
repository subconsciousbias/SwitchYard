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

if [ ! -f "$target" ]; then
  cp "$example" "$target"
  echo "created $target from $example — fill in the blanks"
  exit 0
fi

# Back the file up before touching it at all. Cheap insurance.
backup=".env.backup.$(date +%Y%m%d-%H%M%S)"
cp "$target" "$backup"

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
empty=$(grep -E '^[A-Z_]+=$' "$target" | cut -d= -f1 || true)
if [ -n "$empty" ]; then
  echo
  echo "still empty (fill these in):"
  printf '  %s\n' $empty
fi
