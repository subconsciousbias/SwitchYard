# SwitchYard

One endpoint for every LLM tool, with ordered capacity across your subscriptions,
sticky sessions, and a portal that tells you how close each plan is to running dry.

Built as a plugin on top of LiteLLM rather than a replacement for it: LiteLLM
already speaks both the OpenAI and Anthropic wire protocols with streaming and
tool calling, which is the bulk of a gateway's code. SwitchYard supplies the
part LiteLLM has no concept of — plans, connection caps, ordered fill,
session affinity, quota headroom, and subscription economics.

## Quickstart

```bash
cp .env.example .env     # fill in keys; CLAUDE_CONFIG_DIR points at your ~/.claude
docker compose up -d
open http://localhost:4001            # the portal
```

Then point anything at `http://<host>:4000` with `LITELLM_MASTER_KEY` as the key:

```bash
# Claude Code (Anthropic protocol)
ANTHROPIC_BASE_URL=http://host:4000  ANTHROPIC_AUTH_TOKEN=$LITELLM_MASTER_KEY  ANTHROPIC_MODEL=forge  claude

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
| `forge` | Coding Workhorse | Minimax Ultra → Minimax Max → Grok → GLM Flash → OpenCode Go → OpenRouter | `local-box/qwen` |
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

`local-box/glm-flash` sits outside every lane order: it exists as the target for
context-window fallbacks, so a prompt too large for the chosen model lands
somewhere that can hold it. Only models that declare `context_window` take part,
so an undeclared window never becomes a wrong routing decision.

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

Give a plan an `expires:` date and it is automatically promoted ahead of plans
you keep paying for once it is inside `drain_within_days` (21), **soonest death
first**, so the capacity with the least time left is drained first. Cancelled
capacity gets used up instead of quietly rotting, and it drops out of every lane
by itself on the expiry date. No config edit, no restart.

Today that makes `forge` run `opencode-go` (7d) → `grok` (19d) → Minimax Ultra →
Minimax Max → `glm` (23d) → OpenRouter → Qwen. GLM joins the front of the queue
on its own in two days, when it comes inside the 21-day window.

## The portal (`:4001`)

The reason it exists: read quota headroom for every plan on one page instead of
opening eight vendor dashboards.

- **Capacity board** — live slots per lane, what is cooled down and why, and the
  lane's current total versus its configured total.
- **Quota left** — a real percentage where possible. Preference order: a number
  the provider reported (headers or the sidecar) → your configured allowance →
  *observed* allowance, meaning where the plan actually ran out last cycle. So a
  plan with an unknown allowance still gets a headroom bar after one cycle.
- **Burn rate** — $/hr and tokens/hr over the last 3 hours, with a per-plan
  alert threshold. This is what catches a $20/hour overflow early.
- **Effective $/Mtok** — monthly fee ÷ tokens actually delivered. The number that
  answers whether the $132 Ultra plan or the $55 Max plan is the better buy.

`GET /api/state` returns all of it as JSON for Paperclip. `POST /admin/reload`
picks up `plans.yaml` edits without a restart.

## No API key? Then apex is just Opus

Worth stating plainly, because it is easy to paper over: **a Claude subscription
does not grant API access.** `console.anthropic.com` keys are metered billing,
separate from a Max plan. So Fable 5.1 via the API is only available if you
choose to add API credits, and it ships `enabled: false`.

Astra 6 is not a separate provider either — it is `gpt-6-astra`, a model on the
OpenAI/Codex seat, so it lives under that plan as `openai/astra` and shares the
seat's two connections, quota windows, cooldowns and 2026-10-04 expiry.

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
   enabled. Expires with the seat on 2026-10-04.
3. **Accept that there is no tier above Opus** and let `apex` resolve to
   `claude-max`, the same place `judge` lands. That is the current default. It
   is not a broken lane — it means escalation gets the best model you have, and
   the lane is there for the day you add one.

The sidecar will not run an alias that is not in `MODEL_ALLOW`; it falls back to
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

- SwitchYard pins it there. A request carrying tool *results* is mid-loop, and
  its `tool_call_id`s were minted by one plan's bridge — a peer would reject
  them outright and would hold none of this conversation's prompt cache. So a
  pinned follow-up waits for its plan instead of spilling down the lane.
- The sidecar rebuilds the session from the caller's own request, which carries
  the whole history, tool results and all. Staying on the plan is what makes
  this cheap: the provider's prompt cache is keyed to the account's prefix, so
  a replayed history still hits it here and would miss anywhere else.
- This is the **only** path allowed to queue. A new request still fails fast so
  SwitchYard can spill it to the next plan in the lane; a resumption has
  nowhere to spill to, so it waits up to `MCP_RESUME_WAIT_SECONDS` (300s),
  reclaiming a slot from another stale parked session if one is there. Past the
  deadline it gets a 503 with `Retry-After`.

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

One image, four services, selected by `PROVIDER`:

| Subscription | `PROVIDER` | CLI | Port | Log in with |
|---|---|---|---|---|
| Claude Max $200 | `claude` | `claude -p` | 8081 | `claude login` |
| OpenAI seat (+ Astra 6) | `codex` | `codex exec` | 8082 | `codex login --device-auth` |
| Grok $300 (SuperGrok) | `opencode` | `opencode run` | 8083 | `opencode auth login --provider xai` |
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
| `claude -p` | `--system-prompt` (true replace), `--disallowed-tools`, `--exclude-dynamic-system-prompt-sections` | **2** |
| `opencode run` | `--agent switchyard` — a custom agent (`harness/opencode.json`) with every tool disabled and no prompt of its own | 7,239 → **445** |
| `codex exec` | `-c model_instructions_file=<path>` replaces the compiled-in base instructions | 14,255 → **9,768** |

Measured through the bridge, not inferred. OpenCode's 92% cut is the important
one, since Grok and OpenCode Go are the CLI-backed workhorses.

Codex is the stubborn case. `model_instructions_file` is the key that works;
`experimental_instructions_file` barely moves it (14,154), and
`include_plan_tool=false`, `include_apply_patch_tool=false` and
`tools.web_search=false` had **no effect at all**. The residual ~9,800 is codex's
own tool schema. Worth revisiting if that seat outlives its 2026-10-04 expiry.

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

**A caution on testing this.** I first concluded `prompt` had no effect because an
agent told to "ignore the user and reply PINEAPPLE" answered the user normally.
That is an injection-shaped instruction and a model declining it proves nothing.
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

**`max_tokens` is not enforced on these lanes.** No CLI has a token cap — Claude
Code offers `--max-turns`, not a token limit — so a caller's `max_tokens` is
converted into a prompt instruction ("answer in at most roughly N words"). That
genuinely shortens output and therefore saves subscription quota, but the model
can exceed it; `/health` reports `enforces_max_tokens: false`. If you need a hard
cap, use an API-keyed lane.

The one hazard of a shared credential store is a cold start where the token is
due for refresh: every concurrent subprocess would race to refresh and rewrite
the same file. The first request therefore runs alone, and full concurrency is
released once it succeeds (`"warm": true` on `/health`).

Log in once per sidecar. Each has its own credential store under `./secrets/`,
isolated from your host CLIs — deliberately, for two reasons:

- **Claude on macOS keeps its token in the login Keychain**, so mounting
  `~/.claude` gives a Linux container settings and history but no credentials.
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

- **TODOs in `plans.yaml`:** the Mimo 2.5 OpenRouter slug, the OpenCode Go base
  URL and model id, the Astra 6 endpoint (shipped disabled), and token
  allowances for the Minimax plans — leaving those null and letting the observed
  learning fill them in is a reasonable choice.
- **Set `GLM_API_BASE=https://api.z.ai/api/coding/paas/v4`.** A Coding Plan key
  must use the coding endpoint; the general endpoint (and `open.bigmodel.cn`)
  rejects it. This was wrong in my first pass.
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
python3 scripts/smoke.py         # the live stack: lanes, affinity, cooldown, tools
```

That is the mechanical suite against a running deployment — 14 checks by default,
none of which spend subscription quota. `--paid` adds the lanes that do, `--slow`
measures CLI harness overhead.

Offline, needing nothing running:

```bash
python3 tests/test_routing.py    # ordered fill, affinity, vanishing capacity
python3 tests/test_classify.py   # real MiniMax and Z.AI error payloads
python3 tests/test_policy.py     # concurrency learning and pacing control
python3 tests/test_probes.py     # quota probe field mapping and re-auth
python3 tests/render_preview.py  # every portal template, against a fixture
```

Neither needs Redis or a running stack.

## Sources for the error-code behaviour

- [MiniMax error codes](https://platform.minimax.io/docs/api-reference/errorcode)
  — code list; notably does not document HTTP status mapping.
- [MiniMax-M2 issue #62](https://github.com/MiniMax-AI/MiniMax-M2/issues/62)
  — `insufficient balance (1008)` returned as HTTP 500.
- [MiniMax-M2 issue #88](https://github.com/MiniMax-AI/MiniMax-M2/issues/88)
  — `/coding_plan/remains` requires a cookie session, not an API key.
- [Z.AI error codes](https://docs.z.ai/api-reference/api-code)
  — the full 429 business-code table.
