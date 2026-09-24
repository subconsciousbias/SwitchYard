# SwitchYard

One endpoint for every LLM tool, with ordered capacity across your subscriptions,
sticky sessions, and a portal that tells you how close each plan is to running dry.

Built as a plugin on top of LiteLLM rather than a replacement for it: LiteLLM
already speaks both the OpenAI and Anthropic wire protocols with streaming and
tool calling, which is the bulk of a gateway's code. SwitchYard supplies the
part LiteLLM has no concept of — plans, connection caps, ordered fill,
session affinity, quota headroom, and subscription economics.

**Who this is for:** you pay for several LLM subscriptions and want one endpoint
across all of them, with the spending routed by policy rather than by whichever
one you happened to configure. You need Docker, and at least one plan — every
provider below is optional, including the paid ones.

## Quickstart

```bash
scripts/sync-env.sh      # creates .env (with a random LITELLM_MASTER_KEY) and config/plans.yaml from the examples
$EDITOR .env             # keys for the plans you actually have
$EDITOR config/plans.yaml   # your plans, their limits, and the lane order
docker compose up -d
open http://localhost:4001            # the portal
```

`config/plans.yaml` is yours and is gitignored: it describes what you pay for,
what each plan may spend, and which order the lanes try them in.
`config/plans.example.yaml` is the shipped starting point — nine plans across
seven providers, which you should cut down to the ones you have.

Use `sync-env.sh` rather than `cp .env.example .env`: on an existing checkout
that copy overwrites real credentials, which is the mistake the script exists
to prevent. On a fresh clone `sync-env.sh` also rewrites the copied
`LITELLM_MASTER_KEY=` line with a random `sk-…` key — without that rewrite
the gateway's startup self-check refuses to boot with the published
`sk-switchyard-change-me` placeholder, and `POSTGRES_PASSWORD` keeps its
default.

The gateway (`:4000`) and the portal (`:4001`) bind to **127.0.0.1 by
default**. The gateway authenticates with a single bearer over plain HTTP,
and the portal has no authentication at all, so binding 0.0.0.0 would
expose both to every host on the network. If you want to reach them from
another host on purpose, set `BIND_ADDR=0.0.0.0` in `.env`. See
SECURITY.md for the trust-boundary reasoning.

Then point anything at `http://<host>:4000` with `LITELLM_MASTER_KEY` as the key:

```bash
# Claude Code (Anthropic protocol)
ANTHROPIC_BASE_URL=http://host:4000 \
ANTHROPIC_AUTH_TOKEN=$LITELLM_MASTER_KEY \
ANTHROPIC_MODEL=forge \
ANTHROPIC_DEFAULT_OPUS_MODEL=judge \
ANTHROPIC_DEFAULT_SONNET_MODEL=forge \
ANTHROPIC_DEFAULT_HAIKU_MODEL=bulk \
claude

# Codex / Cursor / Cline / aider / Zed (OpenAI protocol)
OPENAI_BASE_URL=http://host:4000/v1  OPENAI_API_KEY=$LITELLM_MASTER_KEY   # model: forge
```

The model name you ask for is a **lane**, not a provider.

## Plans and models

The structure mirrors what you actually buy:

- A **plan** is a thing you pay for. It owns the credentials, the quota windows,
  the connection limit, the monthly cost and the expiry date.
- A **model** is something a plan serves. It carries no credentials, and has its
  own optional concurrency limit — **a separate ceiling, not a narrowing of the
  plan's.** The plan's limit caps total concurrency across its models; a model's
  caps how much of that one model may take.
- **Lanes reference models**, written `plan/model` — never plans.

Everything shared is shared because it belongs to the plan. Two models on one
plan draw on the same slots, the same quota, the same cooldown and the same
learned concurrency. That is not a special case; it is why `apex` (Claude's heavy
model) and `judge` (Opus) cannot between them open two connections against a
one-connection Claude Max plan:

```
apex took claude-max/fable; judge fell through to openai/sol rather than
double-booking the plan's one connection
```

```yaml
plans:
  local-box:
    max_parallel: 2              # total across every model on this machine
    models:
      qwen:  {model: openai/Qwen3.8-Flash-Next-oQ4e-mtp, max_parallel: 1}
      gemma: {model: openai/gemma-4-26B-A4B-it-oQ4e-mtp, max_parallel: 1}
```

Those are two independent counters, so one qwen and one gemma run together, a
*second* qwen does not, and a third request anywhere on the plan does not either.
The refusal says which limit bit:

```
req -> local-box/qwen
req -> local-box/gemma   skipped=local-box/qwen(model full at 1)
req -> 429: local-box/qwen(model full at 1), local-box/gemma(plan full at 2)
```

Checking a model's limit against the plan's counter — the obvious single-counter
shortcut — would let a plan of 2 with two models at 1 each run only one request.

A lane's advertised capacity accounts for both: per plan, the lesser of the plan's
limit and what its members in that lane can actually reach. `forge` offers 16, not
the 20 its plans' limits sum to, because several members cap themselves lower.

## Lanes

| Lane | Was | Ordered capacity | Tail |
|---|---|---|---|
| `apex` | Judgement — Heavy | `claude-max/fable` → `openai/astra` | `local-box/qwen` |
| `judge` | Judgement — Regular | `claude-max/opus` → `openai/sol` → `glm/glm-5.3` | `local-box/qwen` |
| `forge` | Coding Workhorse | `round_robin:[Minimax Ultra, Minimax Max]` → `weighted:{Minimax Ultra:5, Minimax Max:2}` → `perishable:[Minimax Ultra, Minimax Max]` → Grok → OpenCode Go → OpenRouter | `local-box/qwen` |
| `nest-demo` | Strategy Nesting (demo) | `round_robin:[ lowest_utilization:[Claude Opus, GPT Sol], lowest_utilization:[Claude Fable, GPT Astra] ]` | `local-box/qwen` |
| `local` | Local Only | `local-box/qwen` → `local-box/gemma` | *(none, on purpose)* |
| `bulk` | Basic | `local-box/gemma` → `local-box/qwen` | *(none — already local)* |

A **tail** member is last-resort capacity: it keeps a lane from hard-failing but
never carries normal traffic, and is excluded from the lane's advertised slots.

**A tail must be local.** Its entire job is to still be there once the paid
capacity is exhausted, so a subscription in the tail is self-defeating — it is
precisely what will have run out at the moment the tail is needed. A metered
provider would also do (it fails on money, not quota), but local is preferred.
`load()` rejects a subscription tail outright rather than letting it look fine
until the day it matters.

A happy side effect: because every tail is an API-keyed local model, **no lane
hard-fails on tool calls any more**. A tool-using request that cannot use any
CLI-backed member falls through to local instead of being refused.

The `local` lane has no tail and no cloud members deliberately — when the box is
busy you get a 429 and back off rather than silently spending money.

### Point your client's *light* calls at a cheap lane

Agent clients rarely make one call per turn. Most also fire a small background
request — a conversation title, a summary, a classification — and by default it
goes to whatever model the main turn uses. On a two-connection subscription that
is half your capacity spent on naming the chat, and it shows up as two slots in
use for what looked like one request.

Most clients can be told where to send those. OpenCode takes a `small_model` in
`provider/model` form:

```json
{ "small_model": "switchyard/bulk" }
```

Substitute the provider id you gave SwitchYard. `bulk` and `local` exist for
exactly this: local models, no subscription quota, no competition with real
work for a plan's connections.

Claude Code routes the same calls through three variables that line up with its
built-in haiku/sonnet/opus aliases — `ANTHROPIC_DEFAULT_HAIKU_MODEL`,
`ANTHROPIC_DEFAULT_SONNET_MODEL` and `ANTHROPIC_DEFAULT_OPUS_MODEL`. Those are
what Explore subagents, any agent you tag with `model: haiku` / `sonnet` /
`opus`, and a few background calls actually resolve to; the older
`ANTHROPIC_SMALL_FAST_MODEL` does not cover them. With the three unset the CLI
falls back to built-in ids like `claude-haiku-4-5`, which are not lanes — the
gateway either 400s the request or routes it without going through the lane
picker. Point `ANTHROPIC_DEFAULT_HAIKU_MODEL` at `bulk` (or `local` if your box
is freer than your quota) and the helper agents stop competing with real work
for the subscription's slots.

Context-window fallbacks are a separate mechanism from the lane order: a prompt
too large for the chosen model is handed to the largest-context model available.
Only models that declare `context_window` take part, so an undeclared window
never becomes a wrong routing decision — and if nothing declares a larger window
than the one that was picked, there is no fallback and the request fails
honestly rather than being silently truncated.

## Balancing strategies

A lane's body is a list of entries. Each entry is EITHER a bare `plan/model` ref
OR a single-key mapping `{strategy_name: body}` — a **group** that produces a
visit order on each new-session placement. Ordinary spill-and-fill then runs
through that visit order, so affinity, gating, per-member caps and the
`5h`/target windows all keep working unchanged. The `tail` stays a lane-level
key and is always last under every strategy.

Five strategies are available (see [issue #43](https://github.com/Fledgewing/SwitchYard/issues/43)):

- **`fill`** *(default)* — config order, pure spill-and-fill. The picker never
  reads or writes a Redis key, so flat configs parse bit-for-bit identically to
  the pre-strategy lane format. A lane whose body is only bare refs is
  implicitly a `fill` lane.
- **`round_robin`** — visit the group's members in strict alternation:
  `round_robin: [claude-max/opus, openai/sol]` puts `opus` and `sol` on
  consecutive sessions in turn.
- **`weighted`** — visit by `{ref: weight}` ratio. Weights are integers;
  `weighted: {minimax-ultra/m3: 5, minimax-max/m3: 2}` visits Ultra five times
  for every two visits to Max when neither is full.
- **`lowest_utilization`** — visit the member with the lowest current load
  first. `lowest_utilization` is `perishable` without the hours-to-reset
  signal, so it is the right choice when probes are absent or unreliable
  (e.g. plans whose provider does not publish quota headroom). Like
  `perishable`, the re-rank is **per provider family**: the score never
  competes across families, so the leading family (first appearance in
  the lane body) ranks among itself first and other families sort behind
  among only themselves.
- **`perishable`** — headroom-aware ordering. The portal re-ranks each member
  after every successful probe and writes the visit order to
  `sy:group-order:{gid}`; the picker reads that key on each new session.
  Built for lanes that mix weekly subscriptions whose budget is racing the
  reset. The re-rank is **per provider family**: the score never competes
  across families. A high-room member from another provider cannot rank
  ahead of every member of the leading family; other families keep their
  spill role behind the leading family and rank among only themselves.
  Bucket order follows the first appearance of each family in the lane
  body, so the operator's declared order determines which family leads.

Groups nest recursively up to depth 4, so an outer strategy can dispatch to
inner strategies:

```yaml
order:
  - round_robin:
      - lowest_utilization: [claude-max/opus, openai/sol]
      - lowest_utilization: [claude-max/fable, openai/astra]
```

The outer `round_robin` alternates between the two inner
`lowest_utilization` groups; each inner group visits whichever member is
lighter-loaded at placement time. The picker recurses through the outer
strategy, then the inner one, producing a single visit order per new session.

The cross-cutting rules — affinity wins, per-member gates unchanged,
drain-promotion inside the group, no new model identifiers, hot path cheap,
tail stays last under pacing — apply across every strategy.

## The three behaviours

**Ordered fill.** A lane hands out slots in order: Minimax Ultra's 4, then
Minimax Max's 4, then Grok's 4, and so on. Total lane capacity is the sum of the
live plans' caps. Saturate it and you get a 429 with `Retry-After` instead of a
queue, so clients back off instead of holding workers open.

**Capacity that disappears.** A hard quota rejection cools the plan down and its
slots leave the pool — an 18-slot `forge` drops to 14 the moment Grok runs out,
and traffic starts at the next plan instead. Provider `Retry-After` is honoured
when it looks like a real window reset. See *Reading provider errors* below: this
is the part that is much harder than it looks.

**Session affinity.** A session keeps its provider for 30 minutes of inactivity,
so a long Claude Code or Cursor conversation does not hop mid-task and its prompt
cache stays warm. Clients that can set `X-Session-Id` get exact affinity; for
everything else `switchyard/session.py` fingerprints the conversation prefix,
which is stable across the turns of one session. If a leased plan runs out, the
session re-leases rather than getting stranded on dead capacity.

Verified by `python3 tests/test_routing.py` (no services needed).

## Reading provider errors

`switchyard/classify.py` is the most load-bearing file here, because these
providers' HTTP status codes cannot be trusted. Both of the following are
verified against vendor docs and filed bugs, and both are covered by tests:

**MiniMax reports being out of money as HTTP 500** — `insufficient balance
(1008)` — and its OpenAI-compatible endpoint can also return **HTTP 200** with
the real status in `base_resp.status_code`. A status-code-only classifier scores
the first as a transient blip worth retrying, and the second as a *success*. In
the 200 case a dead plan looks perfectly healthy and keeps receiving every
request the lane can give it. So SwitchYard inspects successful response bodies
too, not just failures.

**Z.AI returns everything as HTTP 429** — ordinary rate limiting (1302),
temporary overload (1305), the current window being used up (1308), the weekly
or monthly quota being gone (1310), the balance being empty (1113), a spend cap
(1316–1321) and *your subscription has expired* (1309, 1314). Same status code,
wildly different correct responses: wait 30 seconds, or sit out the window, or
remove the plan from the config.

So classification goes vendor business code first, HTTP status second, prose
last. Each plan declares a `provider_family` that selects its code table.

Two outcomes beyond the obvious ones:

- **`plan_dead`** (Z.AI 1309/1314, or "subscription expired" in any prose) takes
  the plan out of every lane for 24 hours and tells you to delete it. A dead
  subscription is not a cooldown.
- **`concurrency`** (MiniMax 1041, connection limit) means your `max_parallel` is
  set higher than the plan allows. That is a config bug, not a quota wall, so it
  gets a short sit-out and its own portal warning telling you to lower the cap —
  useful given "probably under 4 connections" was an estimate.

## Learning the real concurrency limit

"Probably under 4 connections" is a guess, and the true number may vary by time
of day. With `concurrency_learning.enabled`, SwitchYard discovers it the way TCP
discovers bandwidth — additive increase, multiplicative decrease:

- a connection-limit refusal (MiniMax 1041, or any provider saying the same in
  prose) **halves** the cap and records the hour it happened in;
- with unmet demand — claims the cap actually denied — and no refusal for a
  while, the cap **creeps up by one**, never past `max_parallel_ceiling`.

Learning is bucketed by hour of day, so a provider that tolerates 6 connections
at 04:00 and 2 at peak settles at both instead of averaging into one wrong
number. An hour's own learning is used once it has `min_samples` of evidence;
until then it inherits the global figure.

Write `max_parallel: auto` to learn from scratch (GLM and OpenCode Go are set
that way, since "severely limited" is not a number), or give a starting figure
plus a ceiling. Claude Max is pinned at `max_parallel_ceiling: 1` so probing can
never touch it.

## Pacing mode — land every subscription at 100%

Off by default; flip it in `plans.yaml` or from the portal button, which
overrides the config at runtime with no restart.

Normally a plan runs flat out, exhausts itself early, and the lane spills onward.
Pacing inverts that: each subscription is throttled so its allowance runs out
*just as the window rolls over*, so you get everything you paid for and nothing
is wasted. If a caller asks for 10 parallel requests and spending 10 would drain
Grok days before its rollover, SwitchYard runs however few the maths allows —
one, if that is what it takes.

Concurrency alone is too coarse a knob for this. At real throughput a single busy
slot can drain a monthly allowance in a day or two, and you cannot go below one
slot. So there are two mechanisms:

**A pace line** — the consumption you should have reached by now,
`allowance × elapsed_fraction × (1 + overshoot)`. While actual consumption is
*above* the line the plan is held closed; it reopens when the line catches up.
That duty-cycles the plan, and the average lands on the line regardless of how
fast individual requests are.

**A throughput cap** for when you are on or behind the line:
`(remaining ÷ seconds_left) ÷ observed_rate_per_slot`, clamped to the learned
limit. Pacing can only ever narrow capacity, never talk you past a provider's
real limit.

`overshoot` (5%) aims deliberately hot so you finish at 100% rather than 95%.

### Windows, and what happens to a cancelled plan

Quota **resets** each window, and unused allowance in a window is gone — it never
carries forward. So pacing is per-window burndown, never an attempt to spread the
total remaining allowance across every window that is left:

```
|---- full month ----|---- full month ----|-- final, 9 days --|
^ spend it all       ^ spend it all       ^ spend it all, faster  X expiry
```

A cancelled plan therefore has several ordinary windows plus one final window
truncated by the expiry date. Each window's deadline is
`min(next_rollover, expiry)`, which means **the final window paces harder** —
same allowance, less time. The portal shows how many full windows remain and how
long the final one is.

### What pacing does not touch

- **The tail is disabled** while pacing. Spilling to a local model would hide the
  fact that you are ahead of budget, and the point is backpressure: a caller
  asking for more than the pace allows gets a 429 and slows down.
- **Metered providers keep fixed caps** — OpenRouter and the Anthropic API have
  no allowance to land on, so they stay on ordinary spill-and-cooldown. Same for
  local models. Override per plan with `pacing: true|false`.
- **A plan with no known allowance cannot be paced.** It says so on the board and
  keeps its fixed cap until the observed-allowance learning has seen one wall.

## Expiring plans get drained first

Give a plan an `expires:` date and it drops out of every lane by itself on that
date. Before then it keeps its configured place in the lane — an expiry date
alone is not a reason to jump the queue, because quota that resets before the
plan dies loses nothing by waiting its turn.

A `fill` lane promotes an expiring plan ahead of its configured order only when
**both** hold for one of its quota windows (5h, weekly, monthly, …):

1. the plan expires before that window rolls over — it is the last window, so
   whatever is left in it is lost rather than reset; and
2. the window is behind its pace line — the share used is less than the share
   of time elapsed between the window's start and the expiry, so at the rate
   normal routing is already achieving it will not drain by itself.

Promoted plans go **soonest death first**; everything else keeps config order.
A window whose usage is unknown never promotes. The log line says why:
`lane=judge -> opencode-go/glm-5.3-flash [...] drain=(weekly final window 20% used, 64% elapsed)`.
`drain_within_days` (21) now only sets when the board starts flagging an
upcoming expiry.

## The portal (`:4001`)

The reason it exists: read quota headroom for every plan on one page instead of
opening eight vendor dashboards.

- **Capacity board** — live slots per lane, what is cooled down and why, and the
  lane's current total versus its configured total.
- **Quota left** — a real percentage where possible. Preference order: a number
  the provider reported (headers or the sidecar) → your configured allowance →
  *observed* allowance, meaning where the plan actually ran out last cycle. So a
  plan with an unknown allowance still gets a headroom bar after one cycle.
- **Burn rate** — $/hr and tokens/hr per model over the last 3 hours, with a
  per-plan alert threshold. This is what catches a $20/hour overflow early.
  (Subscriptions allocate the plan's spend pro-rata by token share; metered
  plans show their own spend.)
- **Effective $/Mtok** — per model on each plan row. For metered plans: the
  model's own spend, shown once it has 1M tokens of its own. For
  subscriptions: the plan's fee ÷ tokens delivered (the token-share
  allocation cancels out, so every model on the plan shows the plan's rate
  once the plan crosses 1M). An em-dash means that threshold is not met yet.

`GET /api/state` returns all of it as JSON for your own client. Burn rate,
monthly tokens/cost, and effective $/Mtok live on each entry of `models[]`
rather than the plan object. `POST /admin/reload` picks up `plans.yaml` edits
without a restart.

## No API key? Then apex is just Opus

Worth stating plainly, because it is easy to paper over: **a Claude subscription
does not grant API access.** `console.anthropic.com` keys are metered billing,
separate from a Max plan. So Fable 5.1 via the API is only available if you
choose to add API credits, and it ships `enabled: false`.

Astra 6 is not a separate provider either — it is `gpt-6-astra`, a model on the
OpenAI/Codex seat, so it lives under that plan as `openai/astra` and shares the
seat's two connections, quota windows, cooldowns and expiry date.

That leaves these options for a tier above Opus:

1. **A heavier model on the Claude Max plan itself**, if your plan exposes one.
   `claude-max/fable` is wired for exactly this — same sidecar, same connection,
   different `--model`. Find the aliases your plan accepts with:

   ```bash
   docker compose exec claude-max-sidecar claude --model bogus 2>&1 | head
   ```

   Then set it as that model's `model:` and flip `enabled: true`. The sidecar
   derives its allowlist from this config, so no `.env` change is needed.

2. **Astra 6 / GPT 6 on the Codex seat** — `openai/astra`, confirmed working and
   enabled. Expires with the seat.
3. **Accept that there is no tier above Opus** and let `apex` resolve to
   `claude-max`, the same place `judge` lands. That is the current default. It
   is not a broken lane — it means escalation gets the best model you have, and
   the lane is there for the day you add one.

The sidecar will not run an alias outside the set it derives from
`config/plans.yaml`; it falls back to
the default and logs loudly, because an `apex` escalation silently served by the
`judge` model is the kind of bug you would never notice.

## Tool capability is a per-plan property

`Plan.can_use_tools` defaults to **true** for every plan, and a plan genuinely
unable to serve a caller's tool definitions sets `supports_tools: false` in
config to say so. The picker still filters on it: a request carrying `tools`
skips any plan marked that way, and a lane with no tool-capable member refuses
with an explanation rather than silently dropping the definitions, which would
look like the model simply choosing not to call anything.

That used to be a blanket rule instead of a per-plan flag, because every OAuth
subscription here was reached through a CLI — `claude -p`, `codex exec`,
`opencode run` — and each of those is **a whole agent harness**, not a model
endpoint. It has its own system prompt, its own tools and its own loop, so a
caller's tool definitions had nowhere to go, the harness's own tools acted on
the sidecar's container rather than the caller's workspace, and the two system
prompts stacked. Every CLI-backed plan was therefore treated as unable to serve
tools, regardless of what it actually supported.

That blanket assumption is going away as each provider's path is fixed
properly, rather than worked around:

- **xAI and OpenAI** move to direct API calls under our own OAuth grant,
  bypassing the CLI harness entirely for these two, so there is no inner loop
  left to fight with the caller's tools.
- **Claude and OpenCode Go** keep the CLI harness but sit behind an MCP bridge
  that inverts it: the harness's own tool call is parked instead of executed,
  and handed back to the caller to run and answer, so the caller's tools do
  reach the model.

Live today: `claude-max` and `opencode-go` run the bridge (`BRIDGE: mcp`) and
serve real `tool_calls`, tool results included, on `apex`, `judge` and `forge`.
`grok` reaches the model through the token proxy instead, which needs no bridge
at all. Only `openai` is still marked `supports_tools: false`: Codex has no MCP
client, so it needs the direct OAuth path.

One thing the OpenCode profile turned on. Its MCP client times a tool call out
after about 60s — a 90s park died with `MCP error -32001: Request timed out` —
which would have capped every tool call the bridge served through it. But
OpenCode calls every tool with `resetTimeoutOnProgress: true`, so
`tool_server.py` sends `notifications/progress` while a call is parked and each
one restarts that timer. The same 90s park then completes. It is sent to any
client that supplies a `progressToken`, so Claude Code benefits too without
needing its own setting.

### A parked session holds a real connection

While the bridge waits for the caller to answer a tool call, the CLI subprocess
stays alive, so a parked session genuinely occupies one of the plan's
connections. That window is also where callers vanish — and behind a proxy we
cannot see them go: LiteLLM holds its own upstream connection open, so the
sidecar's disconnect check (which does work on a direct call) never fires for
anything arriving through the gateway. Two abandoned loops would take a
2-connection plan out of its lane for the full 30-minute idle TTL.

So the sidecar weighs the two claimants instead of waiting: a parked session is
speculative, while the request at the door is real work, and past
`MCP_PARKED_GRACE_SECONDS` (60s) the request preempts the stalest parked
session rather than getting a 429 that SwitchYard would misread as concurrency
pressure and cool a healthy plan for.

Preempting is not dropping the work. The victim's id is remembered, and if its
follow-up does arrive it is **resumed on the same plan**:

- SwitchYard pins it there — but the pin is the session lease, and the lease
  has a TTL (`lease_ttl_seconds`, 1800s). Inside the window the plan still
  holds the provider's prompt cache for this conversation and the loop's quota
  story, so a mid-loop follow-up finishes where it started, spent or not. If
  the pinned plan's slots are all busy, the follow-up waits up to
  `pin_wait_seconds` (10s) for one to free before spilling to a peer — the
  wait holds no slot, so other sessions keep being placed while it does; a
  plan under an active cooldown or a wait of 0 gives way to a peer
  immediately. Past the lease window — by which point no provider's cache is
  warm anyway — the lease is gone and the follow-up places fresh, spilling
  down the lane like any new request.
- The sidecar rebuilds the session from the caller's own request, which carries
  the whole history, tool results and all.
- This is the **only** path allowed to queue. A new request still fails fast so
  SwitchYard can spill it to the next plan in the lane; a resumption has
  nowhere to spill to, so it waits up to `MCP_RESUME_WAIT_SECONDS` (300s),
  reclaiming a slot from another stale parked session if one is there. Past the
  deadline it gets a 503 with `Retry-After`.

### Every lost session resumes, not just preempted ones

Preemption was only ever one way a session could vanish before its follow-up
arrived. The others: the idle reaper collecting it after 30 minutes because the
caller walked away mid-loop (between meetings, a long build, a laptop asleep),
a crashed CLI, the process timeout, a caller the sidecar saw hang up, or a
sidecar restart taking every session with it. All of these used to answer the
late follow-up with a hard 410 that forced the client to start over.

They all rebuild now, by the same mechanism and the same argument: the
request itself carries the whole history, tool results included, so the loop
continues as if nothing happened and the client never learns the session died.
There is no double-execution risk — the caller's tool already ran on its side;
only the delivery of its result is late. What is genuinely lost is the CLI's
own context: the replayed history costs more tokens than answering a live
session would. That is a fair price for never having to walk a client through
recovering client-side. `MCP_REBUILD_LOST=0` restores the old 410.

Rebuild is also what makes the lease expiry safe. Before it existed, a
follow-up that arrived after `lease_ttl_seconds` placed fresh — and if that
put it on a peer plan, the peer's bridge answered 410 to tool_call_ids it had
never minted, and the caller lost the loop anyway. The two now compose: the
lease expiry unpins the stale conversation, and the rebuild lets whichever
plan wins the placement continue it.

### No router-level fallbacks

SwitchYard owns placement, so `router_settings.fallbacks` is empty and
`num_retries` is 0. LiteLLM applies both inside the router, *after* the proxy's
pre-call hook, so every fallback attempt went behind the picker's back: it
skipped the tool-capability filter (a tool request could land on a plan whose
sidecar hard-400s it), claimed no slot, ignored the session lease and the
mid-loop pin, and the success hook booked its tokens against the plan the
*picker* chose — spending one subscription's quota while debiting another's.

It also hid the failures it rescued. A lane listing every member meant a broken
plan was silently retried on a healthy one and looked fine; removing the lists
surfaced `The model xai/grok-4.6 does not exist` on the first try, a stale
OpenCode-style model id that had been masked since the plan moved to the token
proxy.

A failed request now returns to the caller. Its retry re-enters the picker and
gets a fresh pick against the cooldowns the failure just set, which is better
placement than a fixed list can give. Context-window fallbacks stay, as the one
exception: a prompt bigger than the model's window cannot be served where it was
sent at all. `_check_served_deployment` logs loudly whenever the deployment that
answered is not the one that was picked, and books the usage to the plan that
actually spent it.

`num_retries: 0` is non-negotiable in the pinned litellm build. Router's
`async_pre_routing_hook` is the *internal* auto-router hook, dispatched only
inside `async_get_available_deployment` for routing-strategy strategies; it
never iterates registered `CustomLogger` callbacks, so a Switchyard re-pick
there was never going to fire even with `num_retries=1`. A blind router retry
would have re-routed to the same single-member deployment the picker just
chose — exactly the deployment that just 5xx'd — which is why the litellm
version is pinned by tag AND digest and the gateway runs a startup self-check
that catches any future upgrade that moves the self-check to
`async_pre_call_hook`-class dispatch. See the [Litellm pin and self-check](#litellm-pin-and-self-check)
section below.

### Which plan served the request

The response body's `model` echoes what was asked for — usually a lane name
like `judge` — so a caller doing its own token accounting cannot otherwise tell
which subscription the tokens came out of. SwitchYard adds one namespaced key
that a strict client will ignore:

```json
"switchyard": {"lane": "judge", "plan": "claude-max",
               "model": "claude-max/opus", "sticky": true}
```

LiteLLM also puts it in response headers (`x-litellm-model-group`,
`x-litellm-model-name`, `x-litellm-attempted-fallbacks`), which is easy to miss
and lost by any client that keeps only the JSON.

### Litellm pin and self-check

`Dockerfile.gateway` pins the base image by tag AND digest, e.g.
`ghcr.io/berriai/litellm:v1.101.0@sha256:<manifest-digest>`. The tag is what
humans read; the digest is what `docker pull` actually fetches. They MUST move
together — bumping the tag without the digest silently pulls whatever the
registry has at that tag today, and switching the digest (content-addressed)
without the tag is impossible. Upgrading litellm is a deliberate commit that
re-runs the startup self-check on a running image to confirm the contract is
still intact.

Between `gen_litellm` and `exec litellm`, `docker/gateway-entrypoint.sh` runs
`python3 -m switchyard.selfcheck`. The check has two halves:

- **Static AST audit** against the installed litellm source:
  - `Router.async_pre_routing_hook` source contains no reference to any
    `CustomLogger` callback registry (`litellm.callbacks`,
    `litellm._async_success_callback`, etc.). If a future version starts
    dispatching user callbacks here, `num_retries=0` is no longer the whole
    story and the audit fails loudly.
  - `Router.acompletion` reaches `async_function_with_fallbacks`. If the
    retry path stops going through that function, `num_retries` has no
    effect.
  - `async_function_with_retries` short-circuits on `num_retries<=0` with an
    early `raise` BEFORE the retry loop, not via an empty `range(0)` that
    silently re-runs the loop. A regression that gates the whole loop on
    `num_retries > 0` with a fall-through is caught here.
  - `litellm/proxy` references `async_pre_call_hook` somewhere on disk — the
    one hook Switchyard depends on for picker placement.
- **Dynamic loopback probe**: two `http.server` stubs on `127.0.0.1` (one 200,
  one 429 with body `{"detail": "sidecar at capacity (1)"}`); a minimal
  `litellm.Router` (`num_retries=0`, `disable_cooldowns=True`, one deployment
  per stub) and a probe `CustomLogger` registered via
  `litellm.logging_callback_manager`; the success case must reach
  `async_log_success_event` exactly once, the 429 case must reach
  `async_log_failure_event` exactly once, the stub must have received exactly
  one HTTP call (the proof there was no blind retry), and the call must have
  raised `litellm.RateLimitError`.

Any failed check logs `CRITICAL` and exits 1, which the container restart
policy turns into a visible outage rather than a silently broken proxy. There
is no env-var escape hatch: this is a deliberate "fail-loud" gate, run once
per gateway start, whose output is the audit trail in `docker logs`. The
`selfcheck PASSED` line is what a healthy startup looks like.

### `failed to count tokens. Got - 'items'` in the logs

An Anthropic-format request whose tool declares an array property without an
`items` field — `{"name": "...", "input_schema": {"type": "object", "properties": {"tags": {"type": "array"}}}}` is the legal shape, but the OpenAI tool schema requires `items`, and litellm's `_format_type` (inside `litellm/litellm_core_utils/token_counter.py`) was written against the OpenAI shape. It does a bare-subscript `props['items']` in the `type == "array"` branch, which raises `KeyError: 'items'` on first use and the wrapper logs it as `failed to count tokens. Got - 'items'`. Token-counter falls back to a default estimate, but the warning is loud in `docker logs gateway` and the failure path is repeated on every Anthropic tool call.

`Dockerfile.gateway` patches the installed litellm at build time, replacing the bare-subscript with a tolerant `props.get('items') or {'type': 'string'}`, so an items-less array counts as `string[]` instead of raising. `switchyard/selfcheck.py` runs `_audit_token_counter_patch` on every gateway start and prints `ok  litellm._format_type tolerates items-less arrays; items=string still renders 'string[]'` as one of its five audit lines; a missing patch makes that line `CRITICAL` and the container restart loop surfaces it loudly.

A regression after a future litellm pin bump is caught here rather than in production by the **PIN POLICY** at the top of `Dockerfile.gateway`: bumping the tag or digest means re-deriving this patch against the new upstream, and the Dockerfile's exact-occurrence assert fails the build loudly if the new litellm has drifted past the patched line. The Dockerfile header documents the contract; `tests/test_litellm_patch.py` guards it offline (no litellm import).

Where a plan still cannot serve tools — because its path hasn't been fixed yet,
or because it genuinely never will — set `supports_tools: false` on it and the
picker keeps routing tool-using requests around it, the same as it always has.

Two mitigations for the prompt stacking that CLI-backed plans still have on the
non-tool path:

- `SYSTEM_MODE=replace` passes the caller's system prompt with the CLI's
  *override* flag rather than appending to the built-in one, so only one agent
  prompt is in play. Check the flag exists on your version first
  (`claude --help | grep system-prompt`) — a wrong flag is a hard error.
- `BARE=1` (the default) strips the inner harness's tools and caps it at one
  turn, which is as close to a plain completion as a CLI gets.

Where a CLI has no system-prompt flag at all (OpenCode, Codex as configured),
the caller's system text is folded into the top of the prompt rather than
discarded.

## OAuth subscriptions (Claude Max, the OpenAI seat, SuperGrok, OpenCode Go)

LiteLLM authenticates with static API keys and has no OAuth flow at all, so
neither a Claude Max subscription nor a ChatGPT seat can be a deployment.
`sidecars/cli_bridge` handles both the only durable way: the vendor's own CLI
stays the client and owns login and token refresh, and the sidecar exposes it as
an OpenAI-compatible endpoint on the internal network. LiteLLM never sees a
subscription credential.

One image, three services, selected by `PROVIDER`:

| Subscription | `PROVIDER` | CLI | Port | Log in with |
|---|---|---|---|---|
| Claude Max | `claude` | `claude -p` | 8081 | `claude login` |
| OpenAI seat | `codex` | `codex exec` | 8082 | `codex login --device-auth` |
| OpenCode Go | `opencode` | `opencode run` | 8084 | `opencode auth login --provider opencode-go` |

Note that `opencode` and `opencode-go` are *different* providers in OpenCode —
free/community models versus your paid Go plan — so both the login provider id
and the model string need the hyphenated form for that plan.

Device-code flows, not browser callbacks: a callback listener started inside the
container sends your host browser to a `localhost` port that resolves to your Mac,
so it never completes.

### One container per subscription, not one per request

Concurrency is N CLI **subprocesses inside one container**, not N containers. So:

- **one login per sidecar covers every concurrent run** on that subscription;
- the login persists in a host bind mount, surviving restarts, `compose down/up`
  and image rebuilds — it is genuinely one-time;
- the connection limit is enforced by a gate in that one process, which is also
  what makes "refuse immediately when full" possible, so SwitchYard can spill to
  the next plan instead of holding a worker open.

### The CLI versions are pinned

`Dockerfile.sidecar` pins all three: `@anthropic-ai/claude-code@2.1.278`,
`@openai/codex@0.155.1`, `opencode-ai@1.18.31` (OpenCode v1, which is current —
there is no 2.x release, only a `tui-v2` snapshot tag).

They are pinned because the bridge depends on the exact behaviour of each: the
flags it accepts, the JSON event shape it emits, what its MCP client names a
tool, and how long that client will hold a tool call open. An unpinned
`npm install -g` re-resolves to whatever is newest at build time, so a rebuild
for an unrelated reason could swap the CLI under a verified profile — which is
precisely how a `starlette` bump silently broke the portal. Raise a pin
deliberately, then re-run the sidecar checks in `TESTING.md`.

### Keeping the CLI's own prompt small

This matters more than it looks: high-volume work goes to subscriptions, and a CLI
harness injects its own system prompt and tool schema into *every* call. Left
alone that is thousands of tokens of your quota per request, spent on instructions
you did not write.

Each CLI offers a different lever, all now applied:

| CLI | Mechanism | Prompt tokens, trivial call |
|---|---|---|
| `claude -p` | `--system-prompt` (true replace), `--tools ""` (no built-ins), `--strict-mcp-config`, `--setting-sources ""`, `--exclude-dynamic-system-prompt-sections` | **2** |
| `opencode run` | `--agent switchyard` — a custom agent (`harness/opencode.json`) with every tool disabled and no prompt of its own | 7,239 → **445** |
| `codex exec` | `-c model_instructions_file=<path>` replaces the compiled-in base instructions | 14,255 → **9,768** |

Measured through the bridge, not inferred. OpenCode's 92% cut is the important
one, since Grok and OpenCode Go are the CLI-backed workhorses.

Codex is the stubborn case. `model_instructions_file` is the key that works;
`experimental_instructions_file` barely moves it (14,154), and
`include_plan_tool=false`, `include_apply_patch_tool=false` and
`tools.web_search=false` had **no effect at all**. The residual ~9,800 is codex's
own tool schema. Worth revisiting if that seat is renewed.

**On system prompts specifically**, two of the three take the caller's prompt as a
real override: Claude via `--system-prompt`, and Codex via a
`model_instructions_file` written per request, which replaces the compiled-in base
instructions.

**OpenCode's `agent.prompt` does replace the base prompt** — and whether you want
that depends on the provider, which took three attempts and a read of the source to
establish.

`packages/opencode/src/session/llm/request.ts` makes it a binary switch:

```ts
...(input.agent.prompt ? [input.agent.prompt] : SystemPrompt.provider(input.model)),
```

`SystemPrompt.provider()` picks a base prompt file *by model id* —
`prompt/anthropic.txt`, `gpt.txt`, `default.txt` and so on, each 1,700–2,000
tokens. For a model it matches, setting `prompt` replaces that file and saves most
of it: measured 2,351 tokens without a prompt against 589 with one, in a harness
where a base file applied.

**For `xai/grok-*` no base file matches**, so there is nothing to replace and the
prompt is pure cost. In the real deployment path: 445 tokens with no prompt, 604
with one. So the agent deliberately has **no `prompt` field**. Revisit that if you
ever route an Anthropic- or GPT-family model through OpenCode, where the saving
would invert.

What does *not* work, tested:

| Attempted | Result |
|---|---|
| `--prompt`, `--system`, `--system-prompt` flags | **exit 1**, unknown option. `run.ts`'s `builder()` defines no such flag. |
| `agent.<name>.system` | Not a schema key. Unknown keys are absorbed into `agent.options` and passed to the provider as opaque options — hence "adds ~1,800 tokens, changes nothing". |
| `agent.<name>.instructions` | Same; a known upstream gap. |
| top-level `instructions` | Works as designed, but it *adds* content. Never a suppression mechanism. |
| `~/.config/opencode/prompt/<provider>.txt` shadowing | No such lookup exists. `SystemPrompt.provider()` imports fixed bundled files; there is no runtime path. |

**A caution on testing this.** An early conclusion that `prompt` had no effect
came from an agent told to "ignore the user and reply PINEAPPLE" answering the
user normally. That is an injection-shaped instruction, and a model declining it
proves nothing.
A benign marker ("begin every reply with `[SYD]`") was obeyed immediately. Use a
marker, not a jailbreak, to test whether config reached the model.

**Per-request system prompts.** `--dir <tmpdir>` with a generated `opencode.json`
would give a true per-request override, since `--dir` config is read. Not used: it
measured no token benefit for our providers and adds a temp directory per request.
The caller's prompt is folded into the message instead, which is a role difference
(user rather than system) and the one real inconsistency left against Claude and
Codex.

Nothing in config can suppress the `<env>` block (cwd, git state, platform, date)
or a discovered `AGENTS.md` / `~/.claude/CLAUDE.md`. The OpenCode sidecars do not
mount `~/.claude`, so that file is not picked up here — worth knowing it would be.

**`max_tokens` is honored post-hoc on most of these lanes, refused on others.**
None of the CLIs has a token-cap flag — Claude Code offers `--max-turns`, not a
token limit — so the sidecar either truncates the answer to roughly
`max_tokens * 4` UTF-8 bytes and reports `finish_reason: "length"`, or refuses
the request with HTTP 400, depending on the lane:

- **Claude (`claude -p`) and OpenCode (`opencode run`)** honor `max_tokens`
  post-hoc: the sidecar truncates the answer to about `max_tokens * 4` bytes,
  sets `finish_reason: "length"` on the cut, and reports the same billed
  `completion_tokens` the CLI returned. The cut is at a byte boundary, so a
  multi-byte character that crosses it is dropped silently; the model already
  paid for the discarded tokens. Omitting `max_tokens` leaves the length to the
  provider.
- **Codex (`codex exec`)** refuses with HTTP 400 `max_tokens_unenforceable`
  (`error.type: "max_tokens_unenforceable"`, message: `"max_tokens is not
  enforceable on this lane"`). `codex exec` runs its own multi-turn agent loop,
  so cutting its narration mid-stream is not an honest token cap: the CLI
  will keep charging tokens to refill the cut on the next turn.

`/health` reports the mode per lane: `enforces_max_tokens: true` on the
honoring lanes, `false` plus `enforces_max_tokens_reason:
"no-truncation-flag"` on codex. API-keyed lanes are unaffected.

The one hazard of a shared credential store is a cold start where the token is
due for refresh: every concurrent subprocess would race to refresh and rewrite
the same file. The first request therefore runs alone, and full concurrency is
released once it succeeds (`"warm": true` on `/health`).

Log in once per sidecar. Each has its own credential store under `./secrets/`,
isolated from your host CLIs — deliberately, for two reasons:

- **Claude on macOS keeps its token in the login Keychain**, so mounting
  `~/.claude` gives a Linux container settings and history but no credentials.
  On Linux the token sits in the user's keyring (GNOME Keyring / KWallet, read
  by the CLI via `secret-tool`); on Windows it lives in Windows Credential
  Manager, encrypted with DPAPI. None of those stores are reachable from
  inside the container.
- OAuth refresh needs write access, so sharing means a containerised CLI writing
  into the config directory your interactive CLI is using, mid-session.

Two logins on one account are independent; one refresh-token chain copied into
two places can rotate out from under the other. `secrets/README.md` covers
sharing host directories anyway, which does work for the file-based CLIs.

**Connection limits are not set in the compose file.** Each sidecar reads
`config/plans.yaml` itself, takes the tightest `max_parallel` among the plans
sharing its `SWITCHYARD_SUBSCRIPTION`, and derives its allowed model aliases from
those plans' deployments. Change a limit in one place and the sidecar picks it up
within 30 seconds — no restart, no duplicated number to forget.

Two notes on the OpenCode pair. Grok is reached through OpenCode logged in to
xAI rather than through Grok Build, which requires SuperGrok *Heavy*
specifically — and either way a SuperGrok subscription rides xAI's CLI proxy on
a quota entirely separate from metered `api.x.ai` credits. And although both
plans use the same CLI and the same credential directory, they run as **separate
processes**, because their quotas are separate: one process would put two
subscriptions behind a single connection gate.

The sidecar's critical job is mapping "usage limit reached" to **HTTP 429 with
Retry-After**, because that is the signal the cooldown logic keys off. It also
returns 429 rather than queueing when its slots are busy, so the lane spills
instead of blocking a worker.

## Everything lives in `config/plans.yaml`

Caps, cost, expiry, quota model, lane order, and credentials are all in that one
file; the LiteLLM config is generated from it at container start
(`python -m switchyard.gen_litellm`). There is no second file to keep in sync.

### Reloading after you edit `config/plans.yaml`

Three things read that file, and they pick up changes differently:

```bash
scripts/reload.sh        # validate, restart the gateway only if it must, refresh the board
```

It checks the file before touching anything — a syntax error or a lane with no
live members is reported while the old config is still running, rather than
after the gateway has restarted onto it. Then it compares the edited file's
router signature against the one the gateway publishes in Redis, and does what
each component needs:

```bash
docker compose restart gateway                     # routing: only when the router cannot follow
curl -X POST http://localhost:4001/admin/reload    # the portal's board
# sidecars re-read plans.yaml themselves, within 30s
```

**Most edits need no restart.** The gateway watches `plans.yaml` and hot-swaps
a policy-only change — caps, `max_parallel`, quota windows, lane order,
settings, pacing, costs, expiry — into its live registry within ~5 seconds,
with no dropped requests. The exception is the surface LiteLLM bakes into its
router at container start: model strings, `api_base`, credentials, context
windows and the set of lanes itself. A new model or a different credential
cannot take effect in a running router, so `reload.sh` sees the signature
differ and restarts the gateway only then — rather than risking a half-reload
that leaves the router stale, where a plan you thought you had removed would
keep serving.

The portal is a separate process with its own copy, so `/admin/reload` keeps its
board honest without touching traffic. The sidecars re-read `plans.yaml` on a
30-second cache, so a concurrency or model-alias change lands on its own.

Nothing is lost by restarting: learned concurrency, pacing state, cooldowns,
session leases and usage all live in Redis.

### Adding a plan, end to end

Edit `config/plans.yaml`, then run one command:

```bash
scripts/apply.sh
```

It does what `reload.sh` does, plus everything a brand-new plan needs:

1. propagates new `.env.example` keys into `.env` (append-only, always);
2. validates the config before Docker touches anything;
3. rebuilds the baked images **only where their inputs changed** —
   `scripts/image_plan.py` hashes each image's Dockerfile plus every path its
   `COPY` lines name, and compares that with the `switchyard.inputs` label
   stamped on the image when it was built. It prints which files changed. A
   pure `plans.yaml` edit rebuilds nothing, because `./config` is mounted, not
   baked. An image with no label (built before this existed, or by a bare
   `docker compose build`) is rebuilt once;
4. recreates what needs it — containers whose image was rebuilt, that run an
   older image, or whose environment differs from what compose would create
   now (that is how a `.env` edit is caught; values are never printed).
   Services with no model turn running are recreated **together, in one
   compose call**; a sidecar mid-turn is drained in a background job, in
   parallel with the others, and recreated at its next tool-call gap (a
   parked tool-loop session is not a turn: its follow-up is rebuilt from the
   request). Every wait polls its real condition every 2s and moves on the
   moment it holds; a service that never turns healthy fails the apply. Then
   `docker compose up -d --no-recreate` starts anything missing or stopped —
   a new plan's sidecar, a crashed one;
5. runs `reload.sh` (restarts the gateway only if the router cannot follow the
   edit — policy-only changes hot-swap in place — portal board refresh, health
   wait);
6. audits auth per plan and prints the exact sign-in command for each plan
   that is missing one — env keys for `api_key` plans, the credential file
   under `./secrets/` for `cli_sidecar` plans, the OAuth grant for
   `oauth_proxy` plans. It never runs a login itself: those are your
   keychain and browser session. Run it, then `scripts/apply.sh` again.

Useful variants:

```bash
scripts/apply.sh --fast         # build what changed, up -d, exit: no drain, no waits
scripts/apply.sh --plan         # what would be rebuilt/recreated and why; no changes
scripts/apply.sh --dry-run      # validate + audit + report, change nothing
scripts/apply.sh --build        # force a rebuild of every image
scripts/apply.sh --no-build     # never rebuild, even when stale
scripts/apply.sh --skip-reload  # run the reload steps separately
```

## Quota windows: a 5-hour limit *and* a weekly allowance

Most subscriptions enforce several limits at once — a short burst window
(commonly 5 hours) and a weekly allowance, sometimes a monthly one too. Declare
them all and mark exactly one `role: target`:

```yaml
quotas:
  - name: 5h
    role: constraint        # must not overshoot this
    period: rolling_5h
    allowance: null
  - name: weekly
    role: target            # this is the allowance worth filling
    period: week
    allowance: null
```

The target is what pacing tries to fill — **the weekly allowance, nearly always,
not the 5-hour window**. The constraint is a limit to respect, not a goal. Leave
`role` off and the longest period becomes the target.

Pacing then **spends at the slowest rate any window allows**, so the weekly
allowance gets used up while the burst window is never blown:

```
weekly 2.5% used but 5h 95% used  -> binding=5h,     1 slot: pacing weekly (capped by 5h)
fresh 5h window, weekly behind    -> binding=weekly, 2 slots: pacing weekly
5h window spent                   -> 0 slots, weekly untouched at 2.5%
```

Holding is driven only by the *target* window. A constraint window running ahead
of its own line is fine — the rate cap already handles it — so you never stall
needlessly.

**Attributing a wall to the right window.** A provider says "you are out of
quota" without saying which limit you hit, and writing a 5-hour figure into the
weekly window's observed allowance would corrupt every later decision. The reset
time gives it away: a couple of hours means the burst window, a few days means
weekly. With no hint at all, the shortest window gets the blame, since that is
both the likelier culprit and the safer guess.

## Quota headroom, per plan

Only some of these plans will tell you anything, so headroom resolves in
preference order: a number the provider reported → your configured `allowance:` →
*observed* allowance, meaning where the plan actually ran out last cycle. That
last one is why a plan with `allowance: null` still gets a real headroom bar
after one wall.

### Real numbers from a browser session

Three plans publish true headroom — **minimax-ultra**, **minimax-max** and
**opencode-go** — and all three publish it only to a logged-in browser. An API
key gets `cookie is missing (1004)` from MiniMax and `org_required` from
OpenCode, and neither has a documented key-authenticated route. Rather than
settle for estimates, the portal has a **"Real usage"** panel: paste the session
cookie once, and a poller keeps genuine numbers on the board.

Both endpoints below were read from live sessions, so the field paths in
`plans.yaml` describe what they really return:

| Plan | Endpoint | Windows | Unit |
|---|---|---|---|
| minimax-ultra / -max | `platform.minimax.io/backend/account/token_plan/remains_percent` | `5h` + `weekly` | **percent used only** — every `*_count` is `-1` |
| opencode-go | `opencode.ai/console/api/go/status` | `5h` + `weekly` + `monthly` | **dollars**, as microcents ÷ 1e8 |

Two consequences worth knowing. MiniMax returns one entry per model family, so
the path picks `general` by name (`model_remains.model_name=general....`) rather
than by array position, which would shift the day they add a family. And
OpenCode reports *spend against a limit* (`usedMicroCents` / `limitMicroCents`),
so SwitchYard derives `remaining = limit - used`.

#### Setup, in three commands

**1. OpenCode Go needs a workspace id.** Its route answers
`{"code":"org_required"}` without one. In a browser logged in to opencode.ai,
paste this in the devtools console:

```js
await fetch("/console/api/orgs").then(r => r.json())
// [{"id":"wrk_...","name":"Default"}]  <- the id you want
```

**2. Put it in `.env`** — `sync-env.sh` appends the key without touching any
value you have already set:

```bash
scripts/sync-env.sh            # adds OPENCODE_ORG_ID= if missing
# then edit .env:  OPENCODE_ORG_ID=wrk_...
docker compose up -d portal
```

Forget it and the probe names the variable rather than sending an empty header.

**3. Paste each cookie.** These sites use HttpOnly cookies, so
`document.cookie` is *not* enough — the value has to come from a request:
devtools → Network → click any request to that host → copy the entire `Cookie:`
request header. Then in the portal's **Real usage** panel, paste it into that
plan's row and hit **save & test**.

To confirm what the endpoint gives you before involving the portal at all, run
this in the same logged-in browser:

```js
// MiniMax: both windows, percent-used, for the `general` model family
await fetch("/backend/account/token_plan/remains_percent", {credentials: "include"})
  .then(r => r.json())
  .then(d => d.model_remains.find(m => m.model_name === "general"))

// OpenCode Go: all three meters, in microcents
await fetch("/console/api/go/status", {credentials: "include",
                                       headers: {"x-org-id": "wrk_..."}})
  .then(r => r.json()).then(d => d.access.meters)
```

#### Or: run the login ceremony on the host

Pasting the cookie by hand works, but the header is long, easy to truncate, and
puts the operator's hands on raw session material. For plans that opt in
(`probe.login_ceremony: true` plus a matching `probe.login_url` whose host
equals `probe.url`'s — see the commented example under `minimax-ultra` in
`config/plans.example.yaml`), SwitchYard ships a host-side CLI that does the
same thing end to end, with the credential handled by a real browser you own:

```bash
python3 -m switchyard.ceremony minimax-ultra   # or opencode-go, etc.
```

The **Real usage** panel on the portal shows the same command next to each
cookie-probe row, copy-and-paste ready. The CLI opens the plan's login page in
a local Chromium with an ephemeral profile (deleted on exit — nothing persists
to disk), you type username / password / 2FA into the real browser, and on
reaching the post-auth console the CLI POSTs the harvested cookie back to the
existing `POST /admin/probes/{plan}/cookie` endpoint. The portal then runs the
probe, and the new fingerprint shows up in the row.

**Credentials never leave the ceremony.** The CLI never reads, logs or
forwards a password — SwitchYard is the host for the cookie the browser keeps
after login, not for the form fields the browser saw. Cookie domains are
allowlisted against `probe.url`'s host at harvest time, so a cookie that does
not belong to the probe's site is rejected before the POST, and the login URL
is checked at config load against `probe.url` for the same reason. The CLI
refuses plans that have not set `login_ceremony: true`. Paste remains the
fallback for everything else.

#### What you get

- **minimax**: both windows as percentages, e.g. `5h 37% used · weekly 12% used`.
  There is deliberately no token figure — MiniMax publishes none, so the board
  says `reported by provider (% only)` instead of inventing a limit.
- **opencode-go**: three dollar figures against its real caps, e.g.
  `5h $0.00/$12 · weekly $0.01/$30 · monthly $0.10/$60`.
- A window the response omits reports `ok; no data for <window>` rather than
  failing, so one renamed field cannot hide a good reading for another window.
- The **binding** window drives the warning, and it is not always the target: a
  plan at 12% of its weekly allowance but 37% of its 5-hour burst is limited by
  the burst, and the board says so.

Operational properties:

- The cookie is stored in Redis on this host, **never logged, never returned by
  the API** — status shows only a fingerprint like `23 chars ending er=x`.
- When it expires the probe flips to `needs re-auth`, the board says so, and
  polling **stops** until you paste a fresh one. An expired session never
  becomes a request every minute forever.
- Field paths are candidate lists, because vendors rename things. Hit **test
  now** and the panel prints the raw response so you can map the real names, then
  `curl -X POST $PORTAL/admin/reload`.

Worth being clear-eyed about: a session cookie is as powerful as being logged in
— anything the account can do, including billing, it can do. Revoke it by
logging out at the provider, which invalidates the session. If that trade is not
worth it to you, delete the `probe:` block and headroom falls back to ledger
estimates plus observed-allowance learning.

## Before this is live — worth checking

- **Token allowances in `config/plans.yaml`** are left `null` in the example.
  You can fill in a real number per window, or leave them and let the observed
  allowance — where the plan actually ran out last cycle — fill in for you.
- **Set `GLM_API_BASE=https://api.z.ai/api/coding/paas/v4`.** A Coding Plan key
  must use the coding endpoint; the general endpoint (and `open.bigmodel.cn`)
  rejects it — an easy one to get wrong, since the key authenticates against
  both.
- **Codex CLI flags move between releases.** The `codex exec` invocation is
  overridable with `CODEX_ARGS` / `CODEX_EXTRA_ARGS` rather than a code edit;
  check `codex exec --help` if that sidecar 502s.
- **Verify the two sidecars' limit messages once.** The 429 mapping is regex over
  CLI output, so if either vendor rewords "usage limit reached" the lane will see
  a 502 instead. Worth one deliberate exhaustion test each.
- Local models are reached at `host.docker.internal`; set `LOCAL_API_BASE` if
  Ollama/LM Studio is elsewhere.

## Tests

```bash
python3 -m pytest -q             # routing, classification, pacing, probes,
                                 # both bridges, the token proxy
```

No Redis, no Docker, no credentials, and **no provider calls**: `tests/conftest.py`
blocks any socket to a non-loopback address, so a test that reaches for a real
API fails rather than quietly spending your quota. The fake Redis in
`tests/fake_redis.py` stands in for the real one, several tests run stub servers
and real subprocesses on `127.0.0.1`, and every test loads
`config/plans.example.yaml` rather than your own plans — so the result is the
same on any machine, whatever you subscribe to.

Separately, to check a deployment you are actually running:

```bash
python3 scripts/smoke.py         # the live stack: lanes, affinity, cooldown, tools
```

That one is not mocked. It needs the stack up and your credentials present, and
it spends capacity: local models by default, paid quota with `--paid`. 15 checks,
`--slow` adds CLI harness overhead measurement.

`python3 tests/render_preview.py` renders every portal template against a
fixture and writes an HTML file you can open. It is a preview tool rather than a
test — it catches a template that no longer renders, but it does not assert
anything, and `pytest` does not collect it.

## Provenance, and the vendor surfaces this touches

Some of what SwitchYard reads is not in any vendor's public API docs. Where
that is true, here is exactly what it does and how it was established, so you
can judge it rather than take it on trust.

**It never calls a vendor's API with a subscription credential it was not
given.** That is the line, and it is why two obvious shortcuts were rejected.
Anthropic publishes usage at `api.anthropic.com/api/oauth/usage` and the
ChatGPT backend at `chatgpt.com/backend-api/codex/usage`; reaching either from
here would mean presenting a subscription's own OAuth token to an API that
subscription's terms do not cover. Instead:

| What | How it is read | Why that is different |
|---|---|---|
| Claude Max usage | the sidecar runs `claude -p "/usage"` and reads the report the CLI writes to its own transcript | the vendor's client makes its own call, as it does when you type `/usage` |
| OpenAI seat usage | the sidecar reads the `rate_limits` Codex already records in its session rollouts | nothing is called at all; the file is already on disk |
| SuperGrok usage | the token proxy fetches `cli-chat-proxy.grok.com/v1/billing`, the endpoint the Grok CLI reads, with our own OAuth grant | the grant is SwitchYard's, obtained by its own device-code login |
| GLM usage | `api.z.ai/api/monitor/usage/quota/limit` with your API key | an ordinary API-key call |
| MiniMax and OpenCode Go usage | a session cookie you paste, against the same console endpoint your browser calls | you supply the credential, and can revoke it by logging out |

Two further notes on method. The ChatGPT Responses request shape documented in
`sidecars/token_proxy/server.py` was captured by pointing Codex CLI at a local
recorder: nothing was forwarded upstream and no credential was read. It is kept
as the reason that route was **not** taken. And the OpenCode internals described
below come from reading the published `opencode-ai` package, which is
MIT-licensed; the behaviours quoted were confirmed against a live `opencode run`
rather than inferred.

A session cookie is as powerful as being logged in. If pasting one is not a
trade you want to make, delete that plan's `probe:` block and its headroom falls
back to ledger estimates.

## Sources for the error-code behaviour

- [MiniMax error codes](https://platform.minimax.io/docs/api-reference/errorcode)
  — code list; notably does not document HTTP status mapping.
- [MiniMax-M2 issue #62](https://github.com/MiniMax-AI/MiniMax-M2/issues/62)
  — `insufficient balance (1008)` returned as HTTP 500.
- [MiniMax-M2 issue #88](https://github.com/MiniMax-AI/MiniMax-M2/issues/88)
  — `/coding_plan/remains` requires a cookie session, not an API key.
- [Z.AI error codes](https://docs.z.ai/api-reference/api-code)
  — the full 429 business-code table.

## References

- Issue [#43 — Balancing strategies](https://github.com/Fledgewing/SwitchYard/issues/43)
  — design of the nestable strategy groups documented in *Balancing strategies*
  above; covers the four named strategies, nesting depth ≤ 4, and the
  flat-config bit-for-bit invariant.
