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

## Quota headroom, per plan

Only some of these plans will tell you anything, so headroom resolves in
preference order: a number the provider reported → your configured `allowance:` →
*observed* allowance, meaning where the plan actually ran out last cycle. That
last one is why a plan with `allowance: null` still gets a real headroom bar
after one wall.

MiniMax is the awkward one. It *has* an exact endpoint — `/coding_plan/remains` —
but it only accepts a browser cookie session; an API key gets `cookie is missing
(1004)`, and there is no documented API-key alternative. So MiniMax headroom is
ledger-estimated and sharpened by observed-allowance learning. If you want it
exact, the only route is a scraped session cookie, which I would not build until
it is actually annoying.

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
