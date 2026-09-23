# SwitchYard orchestration — how issues land as PRs

This document is the source of truth for the automated pipeline that turns a
SwitchYard GitHub issue into a merged pull request, and for the deploy
handoff that happens afterwards. Operator test steps live in `TESTING.md`;
the agent rules for working in this repo live in `CLAUDE.md`. This file is
the third piece: **how the orchestrator works** and **who is allowed to do
what** during a dispatch.

## How an issue becomes a worktree

The SwitchYard issue watcher (`~/.local/share/switchyard-issue-watcher`)
monitors this repository for new GitHub issues and dispatches each one to
an Orca worktree automatically. By the time you sit down to work on an
issue, the worktree branch and the worktree path may already exist; the
dispatched worker that owns it has either finished, parked, or is still
in flight. Always check `git worktree list` and the worktree's terminal
before starting from scratch.

The watcher hands each issue to the Orca orchestration driver, which
spins up a shared worktree at the same absolute path as the
coordinator's machine and creates one worker terminal per workstream.
Workers edit that one worktree; nothing else.

## The orchestration driver scripts

The orchestration driver scripts live at `~/.config/opencode/scripts/`.
Three scripts, layered:

- **`orchestrate.py`** — deterministic state machine for the Orca
  mailbox. Encodes the command sequences from the version-matched
  Orca guide (`orca skills get orchestration`). Subcommands:
  `bootstrap`, `settle`, `reply`, `release`, `sweep`. Every
  subcommand prints exactly one JSON object on stdout and exits with
  a documented code (`0` ok, `2` Orca command failure, `3` settle
  timeout, `4` inbox needs a human decision).
- **`orchestrate-docker.py`** — thin layer over `orchestrate.py` for
  the case where supervised workers run inside a disposable local
  Docker task container instead of local panes. Adds `provision`,
  `dispatch`, `attach`, `destroy`; the mailbox verbs are unchanged.
- **`orchestrate-auto.py`** — fully scripted replacement for the
  LLM-based orchestration-docker coordinator. Runs the same pipeline
  as the LLM coordinator but as a state machine, with a journal that
  records every phase transition. This is what an auto run executes.

## Phase pipeline

The phases, recorded in the auto journal, are:

```
provisioned -> planned -> wave -> integrated -> review -> parked
(with --merge: parked -> merged -> cleaned)
```

- **provisioned** — task container (or local pane) is paired with the
  shared worktree. The worktree is mounted into every worker terminal.
- **planned** — a planner worker reads the issue, reads the repo, and
  writes `.orca-auto-plan.md` at the worktree root with one
  `### WORKSTREAM` per independent slice. Each workstream owns a
  disjoint `FILES` list so parallel implementers cannot collide.
- **wave** — one or more implementer workers are dispatched in
  parallel, each scoped to a single workstream. Workers edit the
  shared worktree, run the workstream's `VERIFY` commands, and finish
  with `worker_done`.
- **integrated** — an integrator worker commits the union of worktree
  edits onto a feature branch and opens the PR. Push happens here,
  not on a worker's branch in isolation.
- **review** — the pr-review worker inspects the PR and reports a
  verdict. Review cycles are bounded by `--max-review-cycles`; a
  clean review advances to `parked`.
- **parked** — the run is complete and the PR is open. Without
  `--merge`, the run parks with exit code `4` and the PR URL printed
  for an operator to merge manually.

With the explicit `--merge` flag, the pipeline continues:

- **merged** — `gh pr merge` runs (squash by default; `--no-squash`
  produces a regular merge commit). On conflict, a `merge-fix`
  implementer is dispatched to resolve, bounded to three rounds.
- **cleaned** — the task container is destroyed and the
  `/tmp/orca-auto-<task-id>.json` journal is left in place for audit.

## Mailbox verbs (worker side)

Each dispatched worker has a terminal handle (`--from`). The worker
reaches the coordinator through the Orca mailbox:

- **`send --type worker_done`** — terminal-out message for the
  dispatch. `--outcome succeeded` reports the requested work is
  done; `--outcome failed` reports an unrecoverable failure. The body
  is a 3-sentence executive summary: what was done, what was found,
  what is left. Send it exactly once per dispatch; never encode
  failure only in prose and never silently exit. Always include both
  `taskId` and `dispatchId`.
- **`send --type heartbeat`** — liveness signal every 5 minutes while
  actively working. The coordinator uses it to distinguish "still
  thinking" from "hung / crashed". Include both `taskId` and
  `dispatchId` so a straggler heartbeat from a previous dispatch
  cannot mask a hung retry. Skip heartbeats only while blocked
  inside `check --wait` or `ask` — those are themselves liveness
  signals.
- **`ask`** — blocking question to the coordinator. The call durably
  records the question in the dispatch's run and prints the reply
  body when it returns. **Never** use a local TUI prompt for
  interactive questions — the coordinator cannot see it. If `ask`
  times out, resume with the returned message ID instead of
  creating a duplicate.
- **`send --type escalation`** — pre-completion blocker that needs
  the coordinator to act before the worker can continue. Use this
  for a missing dependency, an ambiguous spec, or a guard that has
  fired; do not stall silently.
- **`check --terminal <handle> --json`** — read pending follow-up
  messages from the coordinator. Nothing interrupts a worker: a
  durable message only arrives when you look. Run this at each
  natural checkpoint and once more immediately before
  `worker_done`, so a redirect lands before the task settles.

## Worktree rules

The shared worktree is the container for every dispatch. Three rules
apply to every worker, every time:

1. **Never `docker compose build` / `up -d` / `up -d --force-recreate`
   from a worktree.** `docker-compose.yml` pins `name: switchyard`,
   so every worktree addresses the same compose project: building or
   recreating from a worktree rebuilds and recreates the **live**
   containers from branch code and mounts the worktree's `./config`
   into them. This is not an isolated copy of the stack — it is the
   live stack wearing worktree clothes.
2. **Never merge the worktree branch into `main` from a worktree.**
   The main checkout owns that handoff. Running `git merge` from a
   worktree drags the worktree's branch into the main checkout's
   working tree, which then has to be rebuilt anyway. Just commit;
   let the merge happen at the main checkout (or via the PR).
3. **Never write to `.env`** and never touch credential stores
   (keychain, `docker login`/`logout`, OAuth tokens). `.env` holds
   real API keys; add new keys to `.env.example` only and run
   `scripts/sync-env.sh` to propagate them. Diagnose read-only,
   then hand the user the exact command to run themselves.

Read-only diagnosis from a worktree is fine: `docker compose ps`,
`docker logs`, the portal board. None of those mutate the live
stack.

## After merge: the deploy handoff

**The deploy step is always the operator running
`git pull && scripts/apply.sh --build` from the MAIN checkout.** It
is never a worker, and it is never automated from a worktree.

Why: only the main checkout owns the live stack's baked images.
`switchyard/*.py` is COPYed into the gateway, portal, and sidecar
images; only `./config` is mounted. A worker-built image would
silently replace the running stack and mount a worktree's
`./config` into it — exactly what the worktree rules above exist to
prevent. `docker compose restart` re-runs the OLD code and looks
like the change did nothing; `scripts/apply.sh --build` is what
brings the new code live.

### How an auto `--merge` run should signal a pending deploy

When an auto `--merge` run reaches the `merged` phase, the
orchestrator **should signal a pending deploy** so the operator
notices on their next check-in. The intended mechanism is:

1. **Post a PR comment** on the merged PR. The comment describes
   what shipped and reminds the operator that `scripts/apply.sh
   --build` is the next step from the main checkout. It stays at
   the top of the conversation.
2. **Add the `needs-deploy` label** to the PR (or to the merge
   commit on `main`) at merge time. The label is the operator's
   durable signal that the live stack has not yet picked up the
   change.

The operator removes the `needs-deploy` label **only after**
confirming that the redeploy completed and the self-check
(`docker compose exec gateway python3 -m switchyard.selfcheck`) is
green. Until then the label stays: it is the durable state that
bridges the gap between "merged" and "deployed".

A run that was not started with `--merge` parks instead and the
operator runs the same handoff by hand after merging the PR on
GitHub: `git pull` and `scripts/apply.sh --build` from the main
checkout, then drop the `needs-deploy` label if one was added.

### Not implemented yet — the orchestrator owns this signal

The signal above is owned by the orchestrator itself, which lives
at `~/.config/opencode/scripts/orchestrate-auto.py` (with
`orchestrate.py` and `orchestrate-docker.py` in the same
directory) — a separate codebase outside this repo. The SwitchYard
repo can document the contract; only the orchestrator can
implement it.

As of this writing, `step_merge` in that script calls only
`gh pr view` (to check `mergeable`) and `gh pr merge`; it does
**not** post the comment or add the label. After a successful
merge it sets `state["phase"] = "merged"`, journals the event, and
hands off to `step_cleanup`, which tears down the task container.
Net effect for the operator: an auto `--merge` run reaches
`merged` and immediately advances to `cleaned` with **no signal
left behind**. The live stack stays on the old code and the PR
list is the only durable record that a merge happened.

Until the orchestrator-side change ships, the operator must
watch for merges themselves:

```sh
gh pr list --state merged --search "is:pr author:@me"
```

or use the GitHub notifications feed. Once the orchestrator-side
change lands, the language in this section gets promoted: "should"
becomes "must", and the PR comment + `needs-deploy` label become
the operator's durable signal that a deploy is pending.

## Cross-references

- **`TESTING.md`** — the operator's first-run runbook. Every step
  has a command and what you should see. Now contains only operator
  test steps; orchestrator notes moved here.
- **`CLAUDE.md`** — the agent rules for working in this repo
  (`.env` never written, worktree rules, test placement, the
  `if __name__ == "__main__"` runner trap).
- **Separate guardrails issue** — the repo-level guard that stops
  workers from recreating the gateway they route through lives in
  its own issue and is explicitly out of scope for this doc. The
  orchestrator cannot enforce it; only the repo can. The worktree
  rules above describe the **operator-facing** contract; the
  guardrails issue is the **machine-enforced** sibling.
- **Orchestrator-side deploy-signal change** — the post-merge
  PR comment + `needs-deploy` label that this doc describes as
  "should" lives in `~/.config/opencode/scripts/orchestrate-auto.py`,
  not in SwitchYard. The change that wires `gh pr comment` and
  `gh pr edit --add-label "needs-deploy"` into `step_merge` (and
  creates the label on first use) is tracked outside this repo.
  Until it lands, the "Not implemented yet" subsection above is
  the source of truth for what the operator sees today.
