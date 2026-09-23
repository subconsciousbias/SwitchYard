#!/bin/sh
set -e
PLANS="${SWITCHYARD_PLANS:-/switchyard/config/plans.yaml}"
if [ ! -f "$PLANS" ]; then
  echo "switchyard: $PLANS is missing — is ./config mounted?" >&2
  exit 1
fi
echo "switchyard: generating LiteLLM config from ${PLANS}"
python3 -m switchyard.gen_litellm "$PLANS" /tmp/litellm.generated.yaml
# Fail-loud startup gate: a gateway that will not start is better than one
# that silently mis-routes. The check prints its result on every start, so
# `docker logs` is the audit trail. A CRITICAL exit here is the container's
# way of saying "stop restarting me, the proxy would lie about every request".
# The selfcheck also audits the secrets on which the proxy depends (master
# key length / default placeholder, Postgres password), so a CRITICAL line is
# equally likely to mean "LITELLM_MASTER_KEY is the published default" as
# "the litellm AST drifted".
echo "switchyard: running litellm self-check (env secrets, litellm AST, loopback probe)"
python3 -m switchyard.selfcheck
exec litellm --config /tmp/litellm.generated.yaml --port 4000 --num_workers "${GATEWAY_WORKERS:-2}"
