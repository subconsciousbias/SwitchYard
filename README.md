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
| `forge` | Coding Workhorse | Minimax Ultra → Minimax Max → Grok → GLM → OpenRouter | Qwen local |
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
slots leave the pool — a 15-slot `forge` becomes 11 the moment Ultra runs out,
and traffic starts at Max instead. Provider `Retry-After` is honoured when it
looks like a real window reset. The distinction between "out of quota" (long
sit-out) and "slow down" (seconds) is in `switchyard/classify.py`; it is the most
load-bearing file here, because misreading either way costs you paid capacity.

**Session affinity.** A session keeps its provider for 30 minutes of inactivity,
so a long Claude Code or Cursor conversation does not hop mid-task and its prompt
cache stays warm. Clients that can set `X-Session-Id` get exact affinity; for
everything else `switchyard/session.py` fingerprints the conversation prefix,
which is stable across the turns of one session. If a leased plan runs out, the
session re-leases rather than getting stranded on dead capacity.

Verified by `python3 tests/test_routing.py` (no services needed).

## Expiring plans get drained first

Give a plan an `expires:` date and it is automatically promoted ahead of plans
you keep paying for once it is inside `drain_within_days` (21). Cancelled
capacity gets used up instead of quietly rotting, and it drops out of every lane
by itself on the expiry date. No config edit, no restart.

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

## OAuth subscriptions

LiteLLM authenticates with static API keys and has no OAuth flow at all, so a
Claude Max subscription cannot be a deployment. `sidecars/claude-max` handles it
the only durable way: the Claude Code CLI stays the client and owns login and
token refresh, and the sidecar exposes it as an OpenAI-compatible endpoint on the
internal network. LiteLLM never sees a subscription credential.

The sidecar's one critical job is mapping "usage limit reached" to **HTTP 429
with Retry-After** — that is the signal the cooldown logic keys off. It also
returns 429 rather than queueing when its single slot is busy, so the lane spills
instead of blocking.

Any other OAuth-only plan (a ChatGPT seat via `codex exec`, for instance) follows
the same shape: set `auth: oauth_sidecar` and point its `api_base` at the sidecar.

## Everything lives in `config/plans.yaml`

Caps, cost, expiry, quota model, lane order, and credentials are all in that one
file; the LiteLLM config is generated from it at container start
(`python -m switchyard.gen_litellm`). There is no second file to keep in sync.

## Before this is live — worth checking

- **Confirm quota errors are really 429s.** The classifier keys off status codes
  first and error text second. Minimax and Z.ai are the two I would verify by
  deliberately exhausting a small budget; if either bills out with 402 or a
  200-with-error-body, adjust `classify.py`.
- **TODOs in `plans.yaml`:** the Mimo 2.5 OpenRouter slug, the Astra 6 endpoint
  (disabled until confirmed), the OpenAI model id, and token allowances for the
  Minimax plans (leave them null and let the observed-allowance learning fill
  them in, if you prefer).
- **`OpenCode Go` is absent** — its Sept 7 expiry has passed. Add it back as a
  plan if that is wrong.
- **Whether the OpenAI plan is a seat or API credits.** If it is a ChatGPT seat
  it needs a sidecar like Claude Max, not an API key.
- Local models are reached at `host.docker.internal`; set `LOCAL_API_BASE` if
  Ollama/LM Studio is elsewhere.
