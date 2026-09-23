#!/usr/bin/env bash
# SwitchYard offline test suite. Single entry point: lint-clean and the unit
# suite green, and the portal layout preview rendered.
#
# Exit non-zero on any failure (set -euo pipefail). The two `python3` steps
# are kept separate so a pytest regression is reported on its own line and the
# rendered preview still lands at /tmp/switchyard-preview.html -- the layout
# is the artefact humans check visually, and it should not be lost because
# of a failure two steps earlier.
#
# LITELLM_LOCAL_MODEL_COST_MAP forces litellm to load its bundled cost-map
# backup at import instead of fetching raw.githubusercontent.com -- pytest's
# import-time collection walks every tests/test_*.py module, several of
# which import switchyard.hooks -> litellm, and conftest.py blocks the
# outbound socket (the "tests never call a real provider" policy), so
# without this the collection phase errors out and the runner reports
# "Interrupted: 4 errors during collection" instead of running the suite.
set -euo pipefail

cd "$(dirname "$0")/.."

export LITELLM_LOCAL_MODEL_COST_MAP=True

python3 -m pytest -q "$@"
python3 tests/render_preview.py