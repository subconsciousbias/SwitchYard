# Switchyard

One endpoint for every LLM tool, with ordered capacity across your subscriptions,
sticky sessions, and a portal that tells you how close each plan is to running dry.

Built as a plugin on top of LiteLLM rather than a replacement for it: LiteLLM
already speaks both the OpenAI and Anthropic wire protocols with streaming and
tool calling, which is the bulk of a gateway's code. Switchyard supplies the
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
ANTHROPIC_BASE_URL=http://host:4000  ANTHROPIC_AUTH_TOKEN=$KEY  ANTHROPIC_MODEL=forge  claude

# Codex / Cursor / Cline / aider / Zed (OpenAI protocol)
OPENAI_BASE_URL=http://host:4000/v1  OPENAI_API_KEY=$KEY         # model: forge
```

The model name you ask for is a **lane**, not a provider.

## Lanes

| Lane | Was | Ordered capacity | Tail |
|---|---|---|---|
| `apex` | Judgement — Heavy | Fable 5.1 → Astra 6 | Claude Max |
| `judge` | Judgement — Regular | Claude Max → OpenAI → Grok | Qwen local |
| `forge` | Coding Workhorse | Minimax Ultra → Minimax Max → Grok → GLM → OpenCode Go → OpenRouter | Qwen local |
| `local` | Local Only | Qwen → Gemma | *(none, on purpose)* |
| `bulk` | Basic | Gemma → Qwen | Minimax Max |

A **tail** plan is last-resort capacity: it keeps the lane from hard-failing but
never carries normal traffic, and it is excluded from the lane's advertised slot
count. The `local` lane has no tail and no cloud members deliberately — when the
box is busy you get a 429 and back off rather than silently spending money.

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
request the lane can give it. So Switchyard inspects successful response bodies
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
of day. With `concurrency_learning.enabled`, Switchyard discovers it the way TCP
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
Grok days before its rollover, Switchyard runs however few the maths allows —
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

## OAuth subscriptions (Claude Max and the OpenAI seat)

LiteLLM authenticates with static API keys and has no OAuth flow at all, so
neither a Claude Max subscription nor a ChatGPT seat can be a deployment.
`sidecars/cli_bridge` handles both the only durable way: the vendor's own CLI
stays the client and owns login and token refresh, and the sidecar exposes it as
an OpenAI-compatible endpoint on the internal network. LiteLLM never sees a
subscription credential.

One image, two services, selected by `PROVIDER`:

| Plan | `PROVIDER` | CLI | Port | Concurrency |
|---|---|---|---|---|
| Claude Max $200 | `claude` | `claude -p` | 8081 | 1 |
| OpenAI $100 seat | `codex` | `codex exec` | 8082 | 2 |

Log in once per sidecar — `docker compose exec claude-max-sidecar claude login`,
`docker compose exec codex-sidecar codex login` — or mount host `~/.claude` and
`~/.codex` that are already logged in.

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

MiniMax is the awkward one: it *has* an exact endpoint, `/coding_plan/remains`,
but the endpoint only answers a logged-in browser. An API key gets `cookie is
missing (1004)` and there is no documented API-key route. Rather than settle for
estimates, the portal has a **"Real usage"** panel: paste the session cookie
once, and a poller keeps genuine headroom on the board.

- The cookie is stored in Redis on this host, **never logged, never returned by
  the API** — status shows only a fingerprint like `412 chars ending 9f2a`.
- When it expires the probe flips to `needs re-auth`, the board says so, and
  polling **stops** until you paste a fresh one. An expired session never
  becomes a request every minute forever.
- Field paths in `plans.yaml` are candidate lists, because vendors rename
  things. Hit **test now** and the panel prints the raw response so you can map
  the real field names.

Worth being clear-eyed about: a session cookie is as powerful as being logged in
— anything the account can do, including billing, it can do. Revoke it by
logging out at the provider, which invalidates the session. If that trade is not
worth it to you, delete the `probe:` block and headroom falls back to
ledger estimates plus observed-allowance learning.

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
