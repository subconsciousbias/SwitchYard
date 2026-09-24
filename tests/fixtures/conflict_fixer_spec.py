"""Canonical conflict-fixer spec text (issue #236).

This is the conflict-fixer task spec string produced by
`build_conflict_spec` in `~/.config/opencode/scripts/orchestrate-auto.py`.
It is mirrored here as a Python constant so the in-repo test suite can
pin the contract: any future edit that drops the plumbing fallback
paragraph (`read-tree`, do-not-ask instruction) will fail
`tests/test_orchestrate_conflict_spec.py` and turn CI red.

The fixture is intentionally a hand-typed copy of the live
`build_conflict_spec` body — it is NOT imported from the orchestrator
(the orchestrator lives outside this repo and CI on a fresh clone would
not have access to it). The contract: when the operator updates
`build_conflict_spec` in the orchestrator, this fixture must be
updated to match.

`tests/test_orchestrate_conflict_spec.py::test_live_orchestrator_spec_matches_fixture`
additionally cross-checks against the live orchestrator file when it is
available on the host that runs the test.
"""

# Mirror of the f-string body returned by
# build_conflict_spec(pr_url, base) in
# ~/.config/opencode/scripts/orchestrate-auto.py:1476 (after the
# plumbing-fallback fix for issue #236).
CONFLICT_FIXER_SPEC = """You are a CONFLICT-FIXER in an automated pipeline.

Target: the feature branch behind PR {pr_url} in the shared worktree.
Change: `git fetch origin {base}` then `git merge origin/{base}`; resolve
conflicts one at a time preserving BOTH sides' intent (never delete the other
side wholesale); remove all markers; run the repo's verification; commit and
push. NEVER move or push the base branch; the merge itself happens on GitHub.

If `git merge origin/{base}` is refused by a hook or guard in the worktree
(this checkout is a worktree — `.claude/hooks/guard.sh` denies the merge
verb under `WORKTREE_ONLY`), do NOT ask the runner for permission to proceed:
a worker question cannot be answered in time and stalls the pipeline
(the dispatch settles before the reply arrives and the runner refuses the
late reply as `dispatch_inactive` per `test_refused_reply_continues_pipeline`).
Fall back to the same three-way fold via plumbing instead — `git fetch
origin {base}`; `git read-tree -m -u HEAD origin/{base}` (two-way merge
into the index and working tree); resolve any conflict markers; run the
repo's verification; then `git commit` (this produces a merge commit on
the feature branch); then push as before.
{SPEC_CONTRACT}
Observable acceptance: pushed merge commit; worker_done ends with
CONFLICTS: resolved.
"""
