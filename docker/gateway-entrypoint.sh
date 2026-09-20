#!/bin/sh
set -e
echo "switchyard: generating LiteLLM config from ${SWITCHYARD_PLANS}"
python -m switchyard.gen_litellm "${SWITCHYARD_PLANS}" /tmp/litellm.generated.yaml
exec litellm --config /tmp/litellm.generated.yaml --port 4000 --num_workers "${GATEWAY_WORKERS:-2}"
