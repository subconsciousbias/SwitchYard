#!/bin/sh
set -e
PLANS="${SWITCHYARD_PLANS:-/switchyard/config/plans.yaml}"
if [ ! -f "$PLANS" ]; then
  echo "switchyard: $PLANS is missing — is ./config mounted?" >&2
  exit 1
fi
echo "switchyard: generating LiteLLM config from ${PLANS}"
python3 -m switchyard.gen_litellm "$PLANS" /tmp/litellm.generated.yaml
exec litellm --config /tmp/litellm.generated.yaml --port 4000 --num_workers "${GATEWAY_WORKERS:-2}"
