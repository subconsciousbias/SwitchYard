# Switchyard — first-run testing runbook

Work through this in order. Each step has a command and **what you should see**.
Stop at the first step that does not match, since later steps depend on it.

Set these once in your shell:

```bash
cd ~/Documents/GitHub/switchyard
export GW=http://localhost:4000
export PORTAL=http://localhost:4001
```

For the key, ask the **gateway** rather than parsing `.env` — that is
authoritative, and it sidesteps a real trap: Docker Compose strips inline
`# comments` from `.env` values, while `cut`/`grep` one-liners do not. Parse it by
hand and you can end up sending the key *plus a comment* and getting a confusing
401 from a stack that is working perfectly.

```bash
export LITELLM_MASTER_KEY=$(docker compose exec -T gateway printenv LITELLM_MASTER_KEY | tr -d '\r\n')
```

The name matches `.env`, so if you already export that file into your shell the
commands below work unchanged. Before the stack is up, read it from `.env` with
`sed -n 's/^LITELLM_MASTER_KEY=//p' .env | sed 's/[[:space:]]*#.*//'`.

---

## Run it, do not type it

Everything below that a machine can check is in `scripts/smoke.py`. Prefer it:

```bash
python3 scripts/smoke.py           # free: local and bulk lanes, all behaviour checks
python3 scripts/smoke.py --paid    # also the lanes that spend subscription quota
python3 scripts/smoke.py --slow    # also the CLI harness overhead (minutes)
```

It exits non-zero if anything fails, and every check prints its evidence — the
member that served the request, the slot counts before and after, the tool call
that came back. The prose below explains what each check means and how to
diagnose a failure; it is not a list of commands to retype.

## 0. Offline checks (no credentials needed, ~30 seconds)

```bash
python3 tests/test_routing.py && python3 tests/test_classify.py \
  && python3 tests/test_policy.py && python3 tests/test_probes.py \
  && python3 tests/render_preview.py
```

**Expect:** four "tests passed" lines and a rendered preview. Open
`/tmp/switchyard-preview.html` in a browser to see the portal layout before
anything is live.

---

## 1. Credentials in `.env`

Fill only what you want to test first — a plan with a missing key simply cools
down and the lane moves on, so you can start with one.

| Variable | Where to get it |
|---|---|
| `LITELLM_MASTER_KEY` | Invent one, e.g. `sk-switchyard-` plus random hex. This is what your tools authenticate to Switchyard with. |
| `MINIMAX_ULTRA_API_KEY`, `MINIMAX_MAX_API_KEY` | platform.minimax.io → API Keys. **Use a separate key per plan** so the two plans' quotas stay distinguishable. |
| `MINIMAX_*_API_BASE` | `https://api.minimax.io/v1` (or the `api.minimaxi.com` host if that is what your account shows). |
| `GLM_API_KEY` | z.ai → API keys. |
| `GLM_API_BASE` | **`https://api.z.ai/api/coding/paas/v4`** — a Coding Plan key works only on the coding endpoint. See the warning below; getting this wrong looks exactly like an exhausted plan. |
| ~~`XAI_API_KEY`~~ | **Not needed.** Your SuperGrok subscription is OAuth and runs through OpenCode; metered `api.x.ai` credits are separate billing. |
| `OPENROUTER_API_KEY` | openrouter.ai/keys. Set a spend limit on the key itself as a second line of defence. |
| `ANTHROPIC_API_KEY` | **Not needed.** A Claude subscription does not grant API access, and the metered Fable plan ships disabled. Set this only if you deliberately add API credits. |
| `OPENCODE_DATA_DIR`, `OPENCODE_CONFIG_DIR` | Your existing OpenCode credential paths — default `~/.local/share/opencode` and `~/.config/opencode`. Used by both the Grok and OpenCode Go sidecars. |
| `LOCAL_API_BASE` | Ollama: `http://host.docker.internal:11434/v1`. LM Studio: `...:1234/v1`. |
| `LOCAL_API_KEY` | Your local server's key. If it needs none, put any non-empty string — LiteLLM must send something. |
| `CLAUDE_CONFIG_DIR` | `/Users/temporalis/.claude` — no API key; the sidecar uses your existing login. |
| `CODEX_CONFIG_DIR` | `/Users/temporalis/.codex` — likewise for the ChatGPT seat (and Astra 6 / GPT 6 on it). |

Then confirm the local models are actually reachable from your host:

```bash
export LOCAL_API_KEY=$(sed -n 's/^LOCAL_API_KEY=//p' .env | sed 's/[[:space:]]*#.*//')
curl -s $LOCAL_API_BASE/models -H "Authorization: Bearer $LOCAL_API_KEY" | head -c 400
```

**Expect:** JSON listing your local models. The config is wired for these ids:

| Plan | Model id | Context |
|---|---|---|
| `local-box/qwen` | `Qwen3.8-Flash-Next-oQ4e-mtp` | 262144 |
| `local-box/gemma` | `gemma-4-26B-A4B-it-oQ4e-mtp` | 262144 |
| `local-box/glm-flash` | `GLM-5.3-Flash-oQ4e` | 1048576 |

If your ids differ, fix the `model:` line for that model under its plan in
`config/plans.yaml`. To see the full list with context sizes:

```bash
curl -s $LOCAL_API_BASE/models -H "Authorization: Bearer $LOCAL_API_KEY" \
  | python3 -c 'import json,sys; [print(m["id"], m.get("max_model_len")) for m in json.load(sys.stdin)["data"]]'
```

`local-box/glm-flash` is deliberately **not** in any lane order — it is the target for
context-window fallbacks, so a prompt too large for the chosen plan lands
somewhere that can hold it rather than erroring.

All three are models of the **one** `local-box` plan, so they draw on its single
pool of 2 slots. They run on the same machine; three models at 2 each would put
six concurrent requests on hardware that handles one or two.

`{"error":{"message":"API key required",...}}` means `LOCAL_API_KEY` is unset or
wrong. Without the header it will fail the same way, so keep the `-H` on every
direct call to the local server. (Calls through the gateway carry it for you.)

---

## 2. Bring the stack up

```bash
docker compose up -d --build
docker compose ps
```

**If the gateway build fails with `failed to fetch oauth token: denied: denied`**
on `ghcr.io/berriai/litellm:main-stable`, the image is fine — Docker is sending
broken credentials. A stale or placeholder `ghcr.io` login makes Docker send them
instead of falling back to anonymous, and GHCR refuses. Check what it holds:

```bash
echo ghcr.io | docker-credential-osxkeychain get   # macOS; prints the secret
```

A username like `USERNAME` means a copy-pasted `docker login` placeholder. Clear
it and the anonymous pull works:

```bash
docker logout ghcr.io && docker compose up -d --build
```

Or log in for real if you also pull private images from GHCR:

```bash
echo "<PAT>" | docker login ghcr.io -u <your-github-username> --password-stdin
```

**If a build fails with `no space left on device`**, that is the Docker VM's disk,
not your Mac's. Check and reclaim:

```bash
docker system df                 # what is using it, and what is reclaimable
docker builder prune -f          # build cache only — always safe, just rebuilds slower
docker image prune -f            # dangling (untagged) images only — safe
```

Those two are non-destructive to anything you are running. `docker image prune -a`
removes every image not backing a running container, which will hit your other
projects — only reach for it if the safe prunes are not enough. The whole
Switchyard stack needs roughly 3GB: about 1.7GB for the gateway, 790MB for the
one shared sidecar image and 280MB for the portal.

**Expect:** `redis`, `postgres`, `gateway`, `portal`, `claude-max-sidecar`,
`codex-sidecar` all `running`, with redis/postgres `healthy`.

```bash
curl -s $GW/health/liveliness
curl -s $PORTAL/healthz
```

**Expect:** a liveness response from the gateway, and
`{"ok":true,"plans":12}` from the portal.

```bash
docker compose logs gateway | grep -i switchyard | head
```

**Expect:** `switchyard: 12 plans, lanes=apex,judge,forge,local,bulk`. If this
line is missing, the plugin did not load and **nothing else in this runbook will
behave correctly** — check for an import error above it.

---

## 3. Log the two sidecars in (OAuth plans)

These have no API key. Claude Max and the ChatGPT seat both authenticate through
their own CLI.

Four of your plans have no API key at all; each authenticates through its own
CLI, using a credential store under `./secrets/` that is isolated from your host
CLIs. **Log in once per sidecar** — this is not optional, and mounting your host
directories is not a substitute:

- **Claude on macOS keeps its OAuth token in the login Keychain**, not in a
  file, so there is nothing in `~/.claude` for a Linux container to read.
  Mounting it yields settings and history but zero credentials.

**Use a device-code flow, not the default browser flow.** A plain `codex login`
starts a callback listener *inside the container* and sends your host browser to
`localhost:<port>`, which reaches your Mac, not the container — so it hangs.
`codex login --device-auth` gives you a code to enter on the website instead, with
no callback. (That flag is real but missing from `codex login --help`; it is
accepted.) `opencode auth login --provider <id>` skips the interactive picker,
which also matters when the terminal is a `docker compose exec` pipe.
- Sharing a host directory read-write lets a containerised CLI rewrite the config
  of the CLI you are using interactively, and OAuth refresh *requires* write
  access. See `secrets/README.md`.

If your `.env` pins `CLAUDE_CONFIG_DIR` / `CODEX_CONFIG_DIR` /
`OPENCODE_DATA_DIR` / `OPENCODE_CONFIG_DIR` to host paths, comment those four
lines out to use the isolated stores, then `docker compose up -d` to recreate.

```bash
docker compose exec claude-max-sidecar   claude login
docker compose exec codex-sidecar        codex login --device-auth
docker compose exec grok-sidecar         opencode auth login --provider xai
docker compose exec opencode-go-sidecar  opencode auth login --provider opencode-go

for p in 8081 8082 8083 8084; do
  docker compose exec gateway python -c "
import json,urllib.request
print(json.load(urllib.request.urlopen('http://$(
  case $p in 8081) echo claude-max-sidecar;; 8082) echo codex-sidecar;;
             8083) echo grok-sidecar;; 8084) echo opencode-go-sidecar;; esac
):$p/health')))" 2>/dev/null
done
```

**Expect** each to report its subscription, the concurrency it read from
`plans.yaml`, and the model aliases it will accept, e.g.:

```
{"ok":true,"provider":"claude","subscription":"claude-max","model":"claude-opus-5",
 "models":["claude-opus-5"],"concurrency":1,"in_flight":0}
{"ok":true,"provider":"opencode","subscription":"grok","model":"xai/grok-4",
 "models":["xai/grok-4"],"concurrency":4,"in_flight":0}
```

Then prove each CLI can really authenticate, which the health endpoint cannot
tell you — it only reports configuration:

```bash
docker compose exec -T claude-max-sidecar  claude -p "Reply with exactly: OK" --model opus --max-turns 1
docker compose exec -T grok-sidecar        opencode auth list | grep -E "xAI|credentials"
docker compose exec -T opencode-go-sidecar opencode auth list | grep -E "OpenCode Go|credentials"
```

**Expect** a reply from Claude, and the relevant provider listed for each
OpenCode sidecar (`xAI oauth`, `OpenCode Go api`). `0 credentials` means the login did not persist — check that
`$HOME` inside the container matches where the credential directory is mounted
(`docker compose exec grok-sidecar sh -c 'echo $HOME; opencode auth list'`).

You can also check the harness overhead, which is quota you spend on the CLI's own
prompt rather than your work:

```bash
docker compose exec -T grok-sidecar sh -c \
  'opencode run --model xai/grok-4.6 --format json --agent switchyard "Say OK"' \
  | python3 -c 'import json,sys
for l in sys.stdin:
    e=json.loads(l) if l.strip().startswith("{") else {}
    t=(e.get("part") or {}).get("tokens")
    if t: print("input tokens:", t["input"])'
```

**Expect roughly 425.** Without `--agent switchyard` it is about 7,200 — the
agent's disabled tools are what make a subscription viable for volume. Claude sits
at 2; codex at ~9,800, which is its own tool schema and has resisted every config
key tried.

**This is the check that matters most.** `concurrency` comes from
`config/plans.yaml`, not from the compose file — if it does not match the
`max_parallel` you set, the sidecar could not read the config and is falling back
to a default. And if `models` is missing an alias you expect, that plan is still
`enabled: false`.

---

## 4. Lane-by-lane smoke tests

Start with the cheapest lane and work up. Each call should return a normal
OpenAI-shaped completion.

### 4a. `local` — no cloud spend, proves the plumbing

```bash
curl -s $GW/v1/chat/completions -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H 'Content-Type: application/json' -d '{
    "model":"local",
    "messages":[{"role":"user","content":"Reply with exactly: LOCAL OK"}],
    "max_tokens":200}' | python3 -m json.tool | head -20
```

**Expect:** a completion containing `LOCAL OK`.

The local Qwen is a *reasoning* model, so a small `max_tokens` returns its
thinking truncated mid-sentence with `finish_reason: "length"` — e.g. at 16
tokens you get `We need respond to user: "Reply with exactly: LOCAL OK". Need
final`. That is the model, not the routing. Hence the 200 above.

Then confirm Switchyard routed it, rather than LiteLLM quietly using the lane
alias's default deployment:

```bash
docker compose logs --tail=20 gateway | grep 'switchyard:'
```

**Expect:** `switchyard: lane=local -> local-box/qwen [configured]`. Members are
named `plan/model`: the plan owns the credential, the quota and the connection
limit; the model is what the lane names. The `[...]` is the cap reason —
`configured`, `learned[h14]`, or a pacing decision.

**The definitive check** is Redis, because it cannot be faked by a fallback path:

```bash
docker compose exec -T redis redis-cli -n 1 --scan --pattern 'sy:*'
```

**Expect** keys like these after a request:

```
sy:usage:local-box:d:2026-09-20     usage booked against the PLAN, not the model
sy:pace:local-box                   a per-slot throughput sample for pacing
sy:lease:<keyhash>:fp:<digest>      a session lease, holding a plan/model ref
```

Note that the keys name `local-box`, the plan — not `qwen` or `gemma`. Quota,
slots and cooldowns belong to the plan, so its models share them.

No `sy:*` keys means the plugin is not loaded and every request is bypassing the
slot accounting, affinity and pacing — while still returning 200s.

### 4b. `bulk` — mechanical work

```bash
curl -s $GW/v1/chat/completions -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H 'Content-Type: application/json' -d '{
    "model":"bulk",
    "messages":[{"role":"user","content":"Summarise in one sentence: Switchyard routes LLM requests across several subscription plans, filling each to its connection limit before spilling to the next."}],
    "max_tokens":80}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
```

**Expect:** a one-sentence summary. **Expect in the log:**
`lane=bulk -> local-box/gemma`.

### 4c. `forge` — the workhorse lane, and the important one

```bash
curl -s $GW/v1/chat/completions -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H 'Content-Type: application/json' -d '{
    "model":"forge",
    "messages":[{"role":"user","content":"Write a Python function reverse_words(s) that reverses the order of words in a string. Code only."}],
    "max_tokens":200}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
```

**Expect:** a working function. **Expect in the log:**
`lane=forge -> opencode-go` **or** `-> grok` — *not* `minimax-ultra`. That is the
drain rule working: expiring plans go first. If you skipped OpenCode Go's
credentials, expect `opencode-go(cooled)` in the skipped list and the pick
falling to `grok`.

### 4d. `judge` — one connection, via the sidecar

```bash
curl -s $GW/v1/chat/completions -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H 'Content-Type: application/json' -d '{
    "model":"judge",
    "messages":[{"role":"user","content":"Two sentences: when is a message queue the wrong choice?"}],
    "max_tokens":150}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
```

**Expect:** a considered answer. **Expect in the log:** `lane=judge -> openai`
(the seat expires 2026-10-04, so it drains first), falling back to `grok` then
`claude-max`.

### 4e. `apex` — escalation

**First, find out whether you even have a tier above Opus.** A Max subscription
gives no API access, so `apex` ships resolving to `claude-max` — the same model
`judge` uses. Check what aliases your plan accepts:

```bash
# A heavier tier on the Claude Max subscription, if your plan has one:
docker compose exec claude-max-sidecar claude --model bogus 2>&1 | head -20

# Or Astra 6 / GPT 6, which lives on the Codex seat you already pay for:
docker compose exec codex-sidecar codex --help | grep -A3 -- --model
```

**Expect:** an error listing the valid aliases. If one of them is a heavier tier
than `opus`, wire it up:

1. set `model: openai/<alias>` on that model under its plan;
2. set `enabled: true` on the model;
3. `curl -X POST $PORTAL/admin/reload`.

No `.env` change and no rebuild: the sidecar re-reads `plans.yaml` every 30
seconds and adds the alias to its own allowlist.

Verify the sidecar will actually run it:

```bash
docker compose exec claude-max-sidecar curl -s localhost:8081/health
```

**Expect:** the alias to appear in `"models"`. If it is missing, the sidecar will
run its default instead — and log a warning saying so, because an `apex`
escalation quietly served by the `judge` model is a bug nobody notices.

Then test the lane:

```bash
curl -s $GW/v1/chat/completions -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H 'Content-Type: application/json' -d '{
    "model":"apex",
    "messages":[{"role":"user","content":"One paragraph: the strongest argument against per-window quota pacing."}],
    "max_tokens":200}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
```

**Expect:** a completion, and `lane=apex -> claude-max` in the log — or
`-> claude-max/fable` if you enabled that model above.

Then confirm the shared connection is respected. With the heavy tier enabled,
run an `apex` call and a `judge` call at the same time:

```bash
curl -s $GW/v1/chat/completions -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"apex","messages":[{"role":"user","content":"count to 300 slowly"}],"max_tokens":500}' >/dev/null &
sleep 1
curl -s $GW/v1/chat/completions -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"judge","messages":[{"role":"user","content":"say hi"}],"max_tokens":20}' >/dev/null
wait
docker compose logs --tail=10 gateway | grep -E 'lane=(apex|judge)'
```

**Expect:** the `judge` line shows `claude-max/opus(full at 1)` among its skipped
members and picks something else. Both models share the plan's one connection —
if `judge` had also reached `claude-max`, the sharing is broken and that is a bug
worth reporting.

### 4f. The Anthropic protocol (what Claude Code speaks)

```bash
curl -s $GW/v1/messages -H "x-api-key: $LITELLM_MASTER_KEY" \
  -H 'anthropic-version: 2023-06-01' -H 'Content-Type: application/json' -d '{
    "model":"forge","max_tokens":32,
    "messages":[{"role":"user","content":"Reply with exactly: MESSAGES OK"}]}' \
  | python3 -m json.tool | head -20
```

**Expect:** an Anthropic-shaped response containing `MESSAGES OK`. This proves a
tool that speaks only Anthropic's API can use a lane whose provider is not
Anthropic.

---

## 4g. Tool calls must avoid the CLI-backed lanes

This is the check most likely to matter for Paperclip, since an agent sends tool
definitions on nearly every call.

```bash
curl -s $GW/v1/chat/completions -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H 'Content-Type: application/json' -d '{
    "model":"judge",
    "messages":[{"role":"user","content":"What is the weather in Oslo?"}],
    "tools":[{"type":"function","function":{"name":"get_weather",
      "description":"Get weather for a city",
      "parameters":{"type":"object","properties":{"city":{"type":"string"}},
      "required":["city"]}}}],
    "max_tokens":120}' | python3 -m json.tool | head -30
docker compose logs --tail=5 gateway | grep 'lane=judge'
```

**Expect:** the log line to include `tools` and to pick a member whose plan is
*not* CLI-backed — for `judge` that means `local-box/qwen`, since every other
member of that lane rides a CLI-backed plan. The response should contain a
proper `tool_calls` block.

Then confirm the refusal path is explicit rather than silent:

```bash
curl -s $GW/v1/chat/completions -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H 'Content-Type: application/json' -d '{
    "model":"apex","messages":[{"role":"user","content":"hi"}],
    "tools":[{"type":"function","function":{"name":"noop","parameters":{"type":"object","properties":{}}}}]
  }' | python3 -m json.tool | head -12
```

**Expect:** a 429 whose detail says no plan in the lane can serve tool calls
because every candidate is CLI-backed. That is correct: `apex` is entirely
subscription-backed. **A 200 with no `tool_calls` would be the bug** — it would
mean the definitions were silently dropped.

The practical consequence for Paperclip: send agentic, tool-using work to
`forge`, `bulk` or `local`, and reserve `judge` / `apex` for reasoning calls that
do not carry tools.

## 5. Behaviour tests

### 5a. Ordered fill and total capacity

```bash
for i in $(seq 1 12); do
  curl -s $GW/v1/chat/completions -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
    -H 'Content-Type: application/json' -d '{"model":"forge",
    "messages":[{"role":"user","content":"count to 200 slowly"}],"max_tokens":400}' \
    >/dev/null &
done; wait
docker compose logs --tail=40 gateway | grep 'lane=forge'
```

**Expect:** the first plan's slots filled to its cap, then the next plan, and so
on — never round-robin. Watch it live on the portal's capacity board instead if
you prefer.

### 5b. Session affinity

```bash
for i in 1 2 3; do
  curl -s $GW/v1/chat/completions -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
    -H 'X-Session-Id: affinity-test-1' -H 'Content-Type: application/json' \
    -d '{"model":"forge","messages":[{"role":"user","content":"say hi"}],"max_tokens":10}' \
    >/dev/null
done
docker compose logs --tail=10 gateway | grep 'lane=forge'
```

**Expect:** the first line picks a plan; the next two show the **same plan** with
`(sticky)`. Then repeat without the header and with a different first message —
expect a fresh lease, proving the conversation-prefix fingerprint separates
sessions.

### 5c. Capacity disappearing on exhaustion

You do not need to burn a real quota. Cool a plan down by hand and watch the
lane shrink:

```bash
curl -s $PORTAL/api/state | python3 -c 'import json,sys; d=json.load(sys.stdin); print("forge slots:", [l for l in d["capacity"]["lanes"] if l["lane"]=="forge"][0]["slots_available_now"])'
docker compose exec redis redis-cli -n 1 SET "sy:cool:minimax-ultra" "quota_exhausted|0" EX 120
curl -s $PORTAL/api/state | python3 -c 'import json,sys; d=json.load(sys.stdin); print("forge slots:", [l for l in d["capacity"]["lanes"] if l["lane"]=="forge"][0]["slots_available_now"])'
```

**Expect:** the second number is 4 lower than the first, and a `forge` request
now skips `minimax-ultra(cooled)`. Undo it with the portal's **re-admit** button
or `redis-cli -n 1 DEL sy:cool:minimax-ultra`.

### 5d. Pacing mode

```bash
curl -s -X POST "$PORTAL/admin/pacing?enabled=on" | python3 -m json.tool
```

**Expect:** `{"pacing": true, ...}`, the portal banner switching to "Pacing mode
on", and the tail (`local-box/qwen`) vanishing from the `forge` board.

**Important caveat:** with `allowance: null` on every plan, pacing has nothing to
aim at and will report *"pacing idle: no allowance known"* per plan, leaving caps
alone. That is correct behaviour, not a bug. To see pacing actually bite, put a
real number on one window — e.g. under `minimax-ultra`'s `monthly` window set
`allowance: 200000000` — then `curl -X POST $PORTAL/admin/reload` and watch the
Pacing column. Turn it back off with `?enabled=off` when you are done.

---

## 6. Real MiniMax usage (optional, needs a session cookie)

1. Log in to platform.minimax.io in a browser.
2. Open devtools → Network → any request to the platform → copy the whole
   `Cookie:` request header value. (Or run `document.cookie` in the console,
   which may be missing HttpOnly cookies — if the probe then fails, use the
   Network tab value.)
3. On the portal, **Real usage** panel → paste into the row for the Ultra plan →
   **save & test**.

**Expect one of:**
- `active` with a real "N left" figure — done, the board now shows true headroom.
- `could not find a remaining value` plus a **raw JSON response**. This is the
  likely first outcome, because I could not verify the endpoint's field names.
  Read the raw JSON, then correct the `fields:` paths under that plan's `probe:`
  block in `plans.yaml`, `curl -X POST $PORTAL/admin/reload`, and test again.
- `session rejected (401)` / `needs re-auth` — the cookie was incomplete; use the
  full Network-tab header value.

Remember this cookie is full account access, and logging out at MiniMax revokes
it.

---

## 7. Point Paperclip at it

Change one lane first, not all of them.

```bash
# Paperclip's LLM config, per lane:
#   base_url: http://<switchyard-host>:4000/v1
#   api_key:  $LITELLM_MASTER_KEY
#   model:    forge        (or judge / apex / local / bulk)
```

Send **one** real task through the `forge` lane — something ordinary, like a
small refactor or a status summary on a real issue.

**Expect:**
- the task completes as it did before;
- `docker compose logs gateway | grep lane=forge` shows a single pick;
- the portal's `forge` board shows one slot in use during the call;
- the plan's `This month` token count increases afterwards.

If Paperclip can pass a stable per-task or per-conversation id as
`X-Session-Id`, set it. Affinity then becomes exact instead of inferred from the
conversation prefix, which keeps long tasks on one provider and its prompt cache
warm.

Then move `judge`, then the rest.

---

## A trap worth knowing: GLM's base URL

Verified by direct call with a Coding Plan key:

| Endpoint | Result |
|---|---|
| `https://api.z.ai/api/coding/paas/v4` | **200**, real completion |
| `https://api.z.ai/api/paas/v4` | 429, code 1113 "Insufficient balance or no resource package" |
| `https://open.bigmodel.cn/api/paas/v4` | 429, code 1113 (same, in Chinese) |

So a wrong base URL returns **exactly the signal that means "this plan is
spent"**. Switchyard will correctly classify code 1113 as quota exhaustion and
cool GLM down for 15 minutes, then do it again on the next attempt — a perfectly
reasoned conclusion from a false premise. **If GLM reports quota exhaustion
immediately, check the URL before believing it.**

(`open.bigmodel.cn` is the mainland-China BigModel host, which is a separate
account system from z.ai — not merely a different region of the same one.)

Two related notes:

- **`max_tokens` is advisory on the sidecar lanes.** The CLIs have no token cap,
  so it becomes a prompt instruction rather than a hard limit. An API-keyed lane
  enforces it properly.
- **GLM 4.6 is a reasoning model.** At `max_tokens: 8` it returned
  `finish_reason: length`, empty `content`, and its text in `reasoning_content` —
  the whole budget went on reasoning. The local Qwen behaves the same way. Give
  these plans generous `max_tokens` when smoke-testing, or you will think they
  are broken.
- This also confirms the researched Z.AI error mapping against a live response:
  business code 1113 really does arrive wrapped in an HTTP 429, which is why the
  classifier reads the body rather than trusting the status.

## What I could not verify, and what to watch

These are the parts built from documentation rather than a live call, so check
them first if something misbehaves:

1. **The MiniMax probe endpoint's host, path and field names.** Step 6 exists
   specifically to discover them.
2. **Codex CLI flags.** `codex exec --json --model ... <prompt>` is the assumed
   invocation; flags move between releases. If the `judge` lane 502s on the
   `openai` plan, run `docker compose exec codex-sidecar codex exec --help` and
   set `CODEX_ARGS` in `.env` accordingly.
3. **The two sidecars' limit wording.** The 429 mapping matches phrases like
   "usage limit reached". If either vendor rewords it, an exhausted plan will
   surface as a 502 instead of cooling down. Worth one deliberate exhaustion
   test on each when convenient.
4. **The OpenRouter Mimo 2.5 slug** — confirm at openrouter.ai/models.
5. **Whether your Max plan exposes a model above Opus at all**, and under what
   alias. Step 4e finds out. If it does not, `apex` and `judge` land on the same
   model, which is the honest outcome rather than a misconfiguration.
6. **Whether your Grok plan has API credits or is a chat seat.** If
   `XAI_API_KEY` calls 401, it is a seat and needs a sidecar like the others.
7. **Real token allowances.** Everything works without them, but pacing stays
   idle and headroom stays estimated until either you set them or a plan hits a
   wall once and the observed-allowance learning records it.
