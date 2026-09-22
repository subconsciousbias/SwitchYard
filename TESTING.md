# SwitchYard — first-run testing runbook

Work through this in order. Each step has a command and **what you should see**.
Stop at the first step that does not match, since later steps depend on it.
GitHub issues in this repo are dispatched automatically to Orca worktrees by the SwitchYard issue watcher (~/.local/share/switchyard-issue-watcher), so the branch and worktree for an issue may already exist before you start one.

Set these once in your shell:

```bash
cd /path/to/switchyard
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
| `LITELLM_MASTER_KEY` | Invent one, e.g. `sk-switchyard-` plus random hex. This is what your tools authenticate to SwitchYard with. |
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
| `CLAUDE_CONFIG_DIR` | `$HOME/.claude` — no API key; the sidecar uses your existing login. |
| `CODEX_CONFIG_DIR` | `$HOME/.codex` — likewise for the ChatGPT seat (and Astra 6 / GPT 6 on it). |

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

If your ids differ, fix the `model:` line for that model under its plan in
`config/plans.yaml`. To see the full list with context sizes:

```bash
curl -s $LOCAL_API_BASE/models -H "Authorization: Bearer $LOCAL_API_KEY" \
  | python3 -c 'import json,sys; [print(m["id"], m.get("max_model_len")) for m in json.load(sys.stdin)["data"]]'
```

Context-window fallbacks use whichever configured model declares the largest
`context_window`, so if you run a long-context local model, add it to
`local-box` with its real window and oversized prompts will land there. With
only the two above — both 262144 — there is nothing larger to fall back to.

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
echo ghcr.io | docker-credential-osxkeychain get | \
  python3 -c 'import json,sys; print(json.load(sys.stdin)["Username"])'   # macOS
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
SwitchYard stack needs roughly 3GB: about 1.7GB for the gateway, 790MB for the
one shared sidecar image and 280MB for the portal.

**Expect:** `redis`, `postgres`, `gateway`, `portal`, `claude-max-sidecar`,
`codex-sidecar` all `running`, with redis/postgres `healthy`.

```bash
curl -s $GW/health/liveliness
curl -s $PORTAL/healthz
```

**Expect:** a liveness response from the gateway, and
`{"ok":true,"plans":9}` from the portal.

```bash
docker compose logs gateway | grep -i switchyard | head
```

**Expect:** `switchyard: 9 plans, lanes=apex,judge,forge,local,bulk`. If this
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
docker compose exec opencode-go-sidecar  opencode auth login --provider opencode-go

for p in 8081 8082 8084; do
  docker compose exec gateway python -c "
import json,urllib.request
print(json.load(urllib.request.urlopen('http://$(
  case $p in 8081) echo claude-max-sidecar;; 8082) echo codex-sidecar;;
             8084) echo opencode-go-sidecar;; esac
):$p/health')))" 2>/dev/null
done
```

### The MCP bridge's parking ceiling

`sidecars/mcp_bridge/probe_server.py` exists to measure the one property the
whole bridge rests on: how long an MCP tool call can be held open before the
CLI's own client gives up. Run it against a live CLI whenever you raise a
pinned CLI version:

```bash
docker compose exec -T opencode-go-sidecar sh -lc 'mkdir -p /tmp/mcptest && cat > /tmp/mcptest/opencode.json <<JSON
{"mcp":{"switchyard":{"type":"local","command":["python3","/app/mcp_bridge/probe_server.py"],"enabled":true}},
 "agents":{}}
JSON
cd /tmp/mcptest && opencode run --dir /tmp/mcptest --format json \
  --model opencode-go/glm-5.3-flash \
  "Call slow_echo with text=parked and delay=90. Wait for it, then report what it returned."'
```

**Expect** `status=completed` and `parked (held 90.0s)`. A
`MCP error -32001: Request timed out` means the progress keepalive is not
reaching the client — pass `keepalive=false` in the tool arguments to measure
the raw ceiling deliberately, which is how OpenCode's ~60s limit was found.

SuperGrok is not in that list any more: the `grok` plan is served by
`xai-token-proxy`, which holds SwitchYard's own OAuth grant instead of shelling
out to a CLI. Its grant is taken out **from the host**, because the device flow
needs a human at a browser:

```bash
python3 -m switchyard.oauth login xai          # prints a URL and a code
python3 -m switchyard.oauth status xai         # never prints the token itself
docker compose exec -T xai-token-proxy python3 -c \
  "import json,urllib.request;print(json.load(urllib.request.urlopen(
   'http://localhost:8090/health')))"
```

**Expect** `authorised: true`, `ok: true`, `has_refresh: true`, and
`upstream_base: https://api.x.ai/v1`. The tokens land in `./secrets/oauth.json`
(mode 0600, gitignored), which the container mounts at `/app/secrets` — so the
host CLI and the proxy share one grant and one refresh. `ok: false` with
`authorised: false` is the normal state before the login, not a crash.

**Expect** each to report its subscription, the concurrency it read from
`plans.yaml`, and the model aliases it will accept, e.g.:

```
{"ok":true,"provider":"claude","subscription":"claude-max","model":"claude-opus-5",
 "models":["claude-opus-5"],"concurrency":1,"in_flight":0}
{"ok":true,"provider":"opencode","subscription":"opencode-go",
 "model":"opencode-go/glm-5.3-flash","concurrency":2,"in_flight":0}
```

Then prove each CLI can really authenticate, which the health endpoint cannot
tell you — it only reports configuration:

```bash
docker compose exec -T claude-max-sidecar  claude -p "Reply with exactly: OK" --model opus --max-turns 1
docker compose exec -T opencode-go-sidecar opencode auth list | grep -E "OpenCode Go|credentials"
```

**Expect** a reply from Claude, and `OpenCode Go api` listed. `0 credentials`
means the login did not persist — check that `$HOME` inside the container
matches where the credential directory is mounted
(`docker compose exec opencode-go-sidecar sh -c 'echo $HOME; opencode auth list'`).

You can also check the harness overhead, which is quota you spend on the CLI's own
prompt rather than your work:

```bash
docker compose exec -T opencode-go-sidecar sh -c \
  'opencode run --model opencode-go/glm-5.3-flash --format json --agent switchyard "Say OK"' \
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

Then confirm SwitchYard routed it, rather than LiteLLM quietly using the lane
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
    "messages":[{"role":"user","content":"Summarise in one sentence: SwitchYard routes LLM requests across several subscription plans, filling each to its connection limit before spilling to the next."}],
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

#### Oversized prompt regression (issue #29)

```bash
python3 - <<'PY' | curl -s $GW/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H 'Content-Type: application/json' \
  -d @- | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"][:120] if "choices" in d else d)'
import json
pad = ("Reference text to be summarised in one sentence. " * 2300)
print(json.dumps({"model": "forge", "max_tokens": 100, "messages": [
    {"role": "user", "content": "Summarise in one sentence.\n\n" + pad}]}))
PY
```

**Expect:** a one-sentence answer — HTTP 200, never a 500. A prompt this size
(~110 KB) rides stdin, and an oversized system prompt rides
`--append-system-prompt-file` / `--system-prompt-file`. Before the fix the
prompt was one argv element, `execve` refused it with E2BIG
(`MAX_ARG_STRLEN`, 128 KiB per element), and the client saw a bare 500 that
read as "gateway broken" rather than "prompt too big". Only lanes with
CLI-backed members (`cli_bridge`/`mcp_bridge` plans) ever had the bug;
API-keyed plans post their bodies over HTTP.

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
3. `scripts/reload.sh` — a new model string is router-shaped, so this is the
   one case where the gateway actually restarts.

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

## 4g. Tool capability is a per-plan property

This is the check most likely to matter in practice, since an agent sends tool
definitions on nearly every call. `Plan.can_use_tools` defaults to true; a plan
opts out with `supports_tools: false` in config, and the picker skips a plan
marked that way for a request carrying `tools`.

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

**Expect:** the log line to include `tools`, and the response to contain a
proper `tool_calls` block. Check which member it picked and cross-reference
`config/plans.yaml`: the log line names a `plan/model` ref, and that plan
should not have `supports_tools: false` set.

Every plan serves tool calls. The CLI-backed ones do it through the MCP bridge
(`BRIDGE: mcp` on the three sidecars), which hands the caller's tools to the
vendor's own client and parks its tool call until the caller answers; Grok goes
direct through the token proxy. `supports_tools: false` remains available in
`config/plans.yaml` for a plan that genuinely cannot, and the picker then routes
tool-carrying requests around it — but no plan sets it today.

**A 200 with no `tool_calls` would be the bug**: it would mean the definitions
were silently dropped rather than either served or refused.

### 4h. Vision through the CLI-backed lanes

Image blocks used to be flattened away by the sidecars: a screenshot came
back as a confident near-white hex (e.g. `#EDF6EC` for a magenta swatch --
the model answering from its prior over what such a screenshot "usually"
shows). The fix stages the bytes to disk and tells each CLI how to carry
them: `claude -p` gets `--add-dir` + an explicit `Read` allowlist (and Read
is no longer in its bare-mode disallowed-tools list), `codex exec` gets a
repeatable `-i FILE`, and `opencode run` gets a repeatable `-f FILE`. None
of the CLIs accept raw base64 in the prompt, which is why staging is the
shape they all share.

The test is the magenta-swatch round-trip. Generate the bytes once,
base64 them, and POST through `forge` (which spills to the CLI-backed
lanes -- `claude-max`, `opencode-go`, `codex` -- the way a real agent
would):

```bash
python3 -c "import base64; print(base64.b64encode(open('/tmp/magenta.png','rb').read()).decode())"
```

```bash
read -r B64 < <(python3 -c "import base64; print(base64.b64encode(bytes.fromhex('89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c63f8cf1f7f060006000300013fe46dabe90000000049454e44ae426082')).decode())")

curl -s $GW/v1/chat/completions -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H 'Content-Type: application/json' -d "{
    \"model\":\"forge\",
    \"messages\":[{\"role\":\"user\",\"content\":[
      {\"type\":\"text\",\"text\":\"Reply with only the hex colour of the swatch.\"},
      {\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$B64\"}}]}],
    \"max_tokens\":40}" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
```

**Expect:** a magenta-family hex (`#ff00ff`, `#f0f`, `#cc3399`, ...). A
near-white hex, a refusal, or an "I cannot see images" message are all
failures: the bytes are reaching the model, but the model is not reading
them, which is what the test exists to catch. The same request shape with
`{"type":"image","source":{"type":"base64","media_type":"image/png","data":"..."}}`
is also accepted and should round-trip the same way; the sidecar picks the
shape apart and stages the bytes either way.

A remote `https://...` URL on the image block is intentionally rejected
with `400 images_unsupported` -- the sidecar has no way to fetch it, and
silently dropping it is the bug being fixed.

The per-CLI flag plumbing is verified separately on each sidecar, against
real installed versions, before each release:

```bash
docker compose exec -T codex-sidecar codex exec --json --skip-git-repo-check \
  -i /tmp/magenta.png - <<<'Reply with only the hex colour of the swatch.'

docker compose exec -T opencode-go-sidecar opencode run --format json \
  --model opencode-go/glm-5.3-flash -f /tmp/magenta.png \
  'Reply with only the hex colour of the swatch.'

docker compose exec -T claude-max-sidecar claude -p --output-format json \
  --model opus --max-turns 4 \
  --disallowed-tools 'Bash,Edit,Write,Glob,Grep,WebFetch,WebSearch,NotebookEdit' \
  --add-dir /tmp/magenta-stage --allowed-tools 'Read(/tmp/magenta-stage/**)' \
  'The image is saved as /tmp/magenta-stage/01.png. Read it with your Read tool and reply with only its hex colour.'
```

**Expect** a magenta-family hex from each. The Claude one proves the
`bare_args_images` list is the one in use: Read is unrestricted and the
disallowed list does not contain it.

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

## 6. Usage probes — reading real headroom from a provider's console

**Setup lives in the README** ("Real numbers from a browser session"): which
plans probe, the endpoints, the `OPENCODE_ORG_ID` one-liner, and how to copy each
cookie. This section is only how to check it works.

### 6a. Without any credential

The cheapest check, and the one that runs in CI: both real payload shapes are
served by a stub and driven through the actual `Prober`.

```bash
python3 -m pytest -q tests/test_probes.py
```

**Expect** 9 passed, including `minimax_remains_percent_payload_is_read_as_percentages`
and `opencode_go_status_payload_yields_all_three_meters`. These use the exact
JSON the live endpoints returned, so a parsing regression fails here rather than
in front of you.

### 6b. Against the live endpoints

Paste each cookie per the README, then on the portal's **Real usage** panel hit
**save & test**. Expect, per plan:

- **minimax**: `active`, both windows as percentages, e.g. `5h 37% used ·
  weekly 12% used`. No token figure — MiniMax publishes none.
- **opencode-go**: `active`, three dollar figures, e.g. `5h $0.00/$12 ·
  weekly $0.01/$30 · monthly $0.10/$60`.

Failures and what they mean:

| Panel says | Cause |
|---|---|
| `session rejected (401)` / `needs re-auth` | cookie incomplete — `document.cookie` omits HttpOnly; use the Network-tab value |
| `probe needs OPENCODE_ORG_ID … set in .env` | step 2 of the README setup was skipped |
| `could not find a remaining value` + raw JSON | a field was renamed; map it from the raw response, fix `probe.windows:`, then `curl -X POST $PORTAL/admin/reload` |
| `ok; no data for 5h` | that one window's path is stale — the others still read fine |

### 6c. Confirm it reached the board

```bash
curl -s $PORTAL/api/state > /tmp/state.json
python3 - <<'EOF'
import json
for p in json.load(open("/tmp/state.json"))["plans"]:
    for w in p["quota"]["windows"]:
        print(f"{p['key']:<15} {w['window']:<8} {w['pct_used']}% used   {w['basis']}")
EOF
```

**Expect** one line per probed window. The **binding** window is the one closest
to biting and need not be the target: at 12% weekly but 37% of the 5-hour burst,
the burst is what limits you, and pacing aims at the target while never
overshooting a constraint.

- If you are exercising a lane with strategy groups (`forge`, `nest-demo`), the
  capacity board must show the lane body as nested rows per group — round-robin,
  weighted, perishable and lowest_utilization all render their members under the
  group heading — and the bare `plan/model` entries as a flat row with no
  wrapper. A group body that collapses to one row, or a bare ref that nests,
  means WS2's board rendering did not pick up your config.

---

## 7. Point your client at it

Change one lane first, not all of them.

```bash
# your client's LLM config, per lane:
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

If your client can pass a stable per-task or per-conversation id as
`X-Session-Id`, set it. Affinity then becomes exact instead of inferred from the
conversation prefix, which keeps long tasks on one provider and its prompt cache
warm.

Then move `judge`, then the rest.

---

## 8. Cookie auto-renewal recon (one-time)

Two of the probe-backed plans — **minimax-ultra** / **minimax-max** and
**opencode-go** — authenticate with a session cookie you paste once. The
cookie expires, the probe flips to `needs re-auth`, and you have to paste
another. Whether that ever stops happening is the question this section is
designed to settle, once, by hand.

The recon has three steps. **8a** is the only one that changes SwitchYard's
behaviour — it tells you whether to set `capture_set_cookie: true` under the
plan's `probe:` block in `config/plans.yaml`. **8b** and **8c** are leads for
a future where the probe can be kept alive without you: a refresh route the
console itself calls on page load, or an auth-exchange endpoint the OpenCode
CLI exposes. None of the steps reach a provider from this worktree — they are
all a browser, a clipboard, and a local file.

A working recon record is one row per provider per step. Empty cells mean
"not investigated"; they are not the same as "no" — the operator who skips
a step is the operator who has to come back. **Never paste a cookie value,
refresh token, or `auth.json` content into the recon or anywhere else.**
Fingerprints (`284 chars, #a1b2c3d4`) and filenames are enough.

### 8a. Set-Cookie sliding check (the key evidence)

The probe's `capture_set_cookie` flag (off by default — see
`config/plans.example.yaml` under `opencode-go.probe`) only does anything
useful if the probed endpoint actually returns a fresh `Set-Cookie` on
every successful request. When it does, the probe writes that header into
Redis, the **Real usage** panel on the portal shows the new fingerprint, and
you never have to paste another cookie. When it does not, the flag is a
silent no-op: a stored cookie never gets overwritten and the probe keeps
working until the original cookie's real expiry, at which point it stops
exactly as it does today. So enabling it on a non-sliding endpoint costs
nothing — but it also does nothing, and the answer to "can the cookie
auto-renew" stays "no".

Reproduce the probe's own request, then read what comes back.

**MiniMax** (`platform.minimax.io`):

1. Log in normally in a browser. Devtools open, **Network** tab, "Preserve
   log" on.
2. In the devtools console, run the same call the probe makes:

   ```js
   await fetch("/backend/account/token_plan/remains_percent",
               {credentials: "include"})
     .then(r => ({status: r.status, setCookie: r.headers.get("set-cookie")}))
   ```

3. Click the row for that request in the Network tab. Record three things:
   - **Response status**: expect `200`. A `401` here means the cookie is
     already dead — paste a fresh one (per the README's "Real numbers from a
     browser session") and retry.
   - **Response Headers → `set-cookie`**: is the row present?
     - **Empty / missing** — the endpoint does **not** slide. Record "no
       Set-Cookie" under MiniMax in the table below.
     - **Present, with the same cookie name and a future `Expires` /
       `Max-Age`** — it slides. Record the new `Expires` / `Max-Age` value;
       the probe will keep that rolling window.
     - **Present, with `Expires` in the past or `Max-Age=0`** — that is a
       logout, not a slide. Record "expires in past" and treat it as no
       slide.
4. Application tab → Cookies → `https://platform.minimax.io`. Pick the
   session cookie (the one named in the request's `Cookie:` header — usually
   `session`, `token`, or `acw_tc`). Record its `Expires` / `Max-Age`
   attribute. A cookie with no expiry attribute is a **session cookie**: it
   lives until the browser closes. If step 3 issued a `Set-Cookie`, the
   attribute on the **stored** cookie should reflect the response's
   `Expires` / `Max-Age`; if the two disagree, the server is sliding — note
   both values.

**OpenCode console** (`opencode.ai/console`):

1. Log in normally. Devtools open.
2. The probe sends a header this fetch won't, so add it:

   ```js
   await fetch("/console/api/go/status",
               {credentials: "include",
                headers: {"x-org-id": "wrk_..."}})
     .then(r => ({status: r.status, setCookie: r.headers.get("set-cookie")}))
   ```

   The `wrk_...` value is the workspace id from the README's setup step 1 —
   the route answers `{"code":"org_required"}` without it, which would look
   like a successful response that happened not to slide.
3. Record the same three things: response status, presence of `Set-Cookie`,
   and its `Expires` / `Max-Age`. Same three outcomes as MiniMax.

Outcomes, recorded once per provider:

| Provider | Endpoint | `Set-Cookie` on 2xx? | Expiry shape | Slide? |
|---|---|---|---|---|
| MiniMax | `/backend/account/token_plan/remains_percent` | yes / no | future / past / none | yes / no |
| OpenCode console | `/console/api/go/status` | yes / no | future / past / none | yes / no |

A **yes** in the last column is the only condition under which step 8d
turns the flag on. The fingerprint visible on the portal's **Real usage**
panel (e.g. `284 chars, #a1b2c3d4`) is what tells you the flag is working:
hit **test now** twice in a row on that plan and the fingerprint should
change on the second hit — the probe logged
`cookie rotated by provider (plan=…, fingerprint=…)` on the first and the
second sees a new value.

### 8b. Refresh/token route search

A sliding cookie is the easiest way to keep the probe alive, but not the
only one. Some consoles extend the session out of band — a separate
`/auth/refresh` or `/session/extend` call fired by the page itself on load
— which the probe could in principle fire once before its real call.
Recording whether such a route exists does not change behaviour today; it
is a lead for a future probe that calls it pre-flight.

Per provider, in the same logged-in browser session as 8a:

1. Network tab, filter **Fetch/XHR** (or **All** if you want to see beacons
   too).
2. Hard-reload the console page (`Cmd-Shift-R` / `Ctrl-Shift-R`). Capture
   every request on first paint.
3. Skim the request paths. Candidates worth a closer look, with the shape
   the probe could call:
   - `…/auth/refresh`, `…/token/refresh`, `…/session/extend`,
     `…/token/rotate`, `…/api/v1/auth/…`
   - any request whose response carries a `Set-Cookie` even though it is
     not itself the probe route
4. Record presence / absence per provider:

| Provider | Refresh route present? | Path (if any) | Fires on page load? |
|---|---|---|---|
| MiniMax | yes / no | | yes / no |
| OpenCode console | yes / no | | yes / no |

"yes" on both columns is the pre-flight shape: the probe could call it once
before the real probe request and inherit any new `Set-Cookie`. "yes on
first, no on second" means the route exists but only in response to user
action (e.g. a click), which is not useful for a poller. "no" closes the
lead.

### 8c. OpenCode CLI-token lead

The OpenCode CLI is what the sidecar shells out to (`opencode run --model
…`), and its own OAuth login is what it uses to authenticate — stored
under `./secrets/opencode/data/auth.json` (and the same for `opencode2`),
per `secrets/README.md`. If that CLI exposes an auth-exchange route that
mints or mirrors a console session, the probe could use it instead of a
pasted cookie. Check by inspection, never by running the binary against a
live account from this worktree.

```bash
# Find the binary the sidecar actually runs (read-only, against the live stack):
docker compose exec -T opencode-go-sidecar which opencode

# Copy it out into a scratch dir on the host (NOT the live stack's config):
mkdir -p /tmp/opencode-recon && cd /tmp/opencode-recon
docker compose cp opencode-go-sidecar:$(docker compose exec -T \
  opencode-go-sidecar which opencode) ./opencode

# Inspect without executing. None of these print credential values — they
# look at the binary's own string table for hints about auth endpoints.
go version -m ./opencode 2>/dev/null | head -5     # build info, if a Go binary
strings ./opencode | grep -E "opencode\.ai|/auth/|/token/|/session/" \
  | sort -u | head -40
# Fallback if strings(1) is not installed:
grep -aoE "(/[a-z]+/)?(auth|token|session|refresh)[a-z/_-]{0,40}" \
  ./opencode | sort -u | head -40
```

What you are looking for in the output is any path that looks like a
console auth endpoint — `/api/auth/...`, `/api/v1/auth/...`,
`/backend/account/...`, anything with `session`, `refresh`, `token` in it,
especially if it appears next to a host string (`opencode.ai`,
`api.opencode.ai`). A Go module path that mentions the same host is weaker
evidence but still worth recording. Many of the matches will be unrelated
library strings — triage by what appears *next to* a literal host string,
which is much harder to fake by accident.

Outcomes, recorded once:

| What was found | Path / host / module | Plausibility |
|---|---|---|
| Auth path in `strings` | | high / medium / low |
| Host literal in `strings` | | high / medium / low |
| Build module path mentions console | | high / medium / low |

**A high-plausibility find is the only thing that would justify adding a
second credential flow to the probe.** Even then, it is a feature, not a
bug-fix: the cookie path is already verified end to end, and the right
next move is a separate dispatch for an OAuth-via-CLI prober — not a
config edit to `plans.yaml`.

What you also record but never paste into the recon:

- `ls secrets/opencode/data/auth.json secrets/opencode2/data/auth.json` —
  the files exist or they do not (mode `0600`, gitignored). Their CONTENTS
  are a refresh-token chain and must never be echoed. If a file is missing,
  `scripts/auth_audit.py` will tell you which sidecar has no login.

### 8d. Decision guide: enabling `capture_set_cookie`

| 8a result | 8b / 8c result | Action |
|---|---|---|
| Slide = yes | either or both leads noted | Set `capture_set_cookie: true` under that plan's `probe:` block in `config/plans.yaml` |
| Slide = yes | both empty | Same as row 1 — but record in the recon that the operator did not look for refresh / CLI-token leads, so a future operator knows to revisit |
| Slide = no | either or both leads noted | Do **not** enable. The flag is a silent no-op on a non-sliding endpoint, and leaving it commented out keeps the intent visible |
| Slide = no | both empty | Do not enable. Record the negative slide result so the next operator does not redo 8a |

**Mechanics of turning it on:**

The flag lives on the probe plan you already configured, in your own
`config/plans.yaml` (gitignored — your live config). Uncomment the
`capture_set_cookie: true` line under the plan's `probe:` block — the
example is in `config/plans.example.yaml` under `opencode-go`. Then
rebuild from the main checkout, **not** this worktree:

```bash
# From the main checkout, NOT this worktree:
scripts/apply.sh --build
```

`apply.sh` restarts the gateway and sidecars, which re-reads
`plans.yaml` and picks up the new flag. The fingerprint on the portal's
**Real usage** panel is the proof: hit **test now** twice on that plan
and the second reading's fingerprint differs from the first. If it does
not, the endpoint does not actually slide and the flag should come back
off.

A flag that fires correctly is also the right time to subscribe to the
gateway log line:

```bash
docker compose logs -f gateway | grep 'cookie rotated'
```

A successful probe run with `capture_set_cookie: true` will print one
line per rotation, with `plan=…` and `fingerprint=…` only — never the
cookie value.

---

## A trap worth knowing: GLM's base URL

Verified by direct call with a Coding Plan key:

| Endpoint | Result |
|---|---|
| `https://api.z.ai/api/coding/paas/v4` | **200**, real completion |
| `https://api.z.ai/api/paas/v4` | 429, code 1113 "Insufficient balance or no resource package" |
| `https://open.bigmodel.cn/api/paas/v4` | 429, code 1113 (same, in Chinese) |

So a wrong base URL returns **exactly the signal that means "this plan is
spent"**. SwitchYard will correctly classify code 1113 as quota exhaustion and
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

## 8. Caller environment — issue #44 smoke tests

These verify the per-request resolution of the caller's tool-execution
environment (the bug where a Claude Code tab pointed at the gateway thinks
it is on Linux in `/app/mcp_bridge`). Three live checks; the offline test
suite covers parsers and probe-id mechanics, not the LiteLLM transport.

### 8a. The metadata-transport assumption

The gateway stamps `metadata.switchyard.caller_env` on every request it
processes. Nothing in this repo proves LiteLLM forwards that field to the
sidecar at runtime, so the sidecar must not require it -- the passive
parsers in `switchyard/caller_env.py` are the primary source, the stamp
is belt-and-braces. Prove both code paths land on the same answer:

1. Send a request **without** the metadata field, carrying an OpenCode-
   shaped environment block in the system prompt:
       curl -s -X POST http://localhost:8081/v1/chat/completions \
         -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
         -d '{"model":"judge","messages":[
           {"role":"system","content":"<environment>\n  <working_directory>C:\\\\Users\\\\demo\\\\proj</working_directory>\n  <platform>windows</platform>\n</environment>"},
           {"role":"user","content":"Hi"}]}'
   The sidecar log line `[SwitchYard ... caller_env]` should show
   `source=request`, `platform=windows`, `cwd=C:\Users\demo\proj`. The
   rendered CLI prompt (visible with `--print`/`--verbose` on the pinned
   `claude -p`) must include the `[SwitchYard tool execution environment]`
   block naming those values.
2. Send the same request **with** a stamped metadata field:
       curl ... -d '{"model":"judge","metadata":{"switchyard":{"caller_env":
         {"platform":"macos","cwd":"/Users/demo/proj","shell":"zsh",
          "source":"host"}}},"messages":[...]}'
   The sidecar must honour the stamp and render the macOS values, even
   though the system-prompt block says Windows. The stamp is the easy
   path; this confirms it actually travels.

If (1) works but (2) shows Windows anyway, LiteLLM is dropping the
metadata field somewhere -- a bug, but expected behaviour given the
known assumption. The passive path still produces a correct answer in
that case, which is the resilience contract.

### 8b. Real Claude Code tab — answer reflects the caller's OS

1. Point a Claude Code tab at the gateway (`ANTHROPIC_BASE_URL=http://...
   /v1`, `ANTHROPIC_API_KEY=$LITELLM_MASTER_KEY`), open a session on
   Windows / macOS / Linux, and ask:
       what OS am I on and what folder are we in?
   Expected (Windows host):
       Windows (10/11, exact build), working in C:\Users\<user>\<repo>
   3 out of 3 runs. The old bug returned `Linux (WSL2, kernel ...), working
   in /app/mcp_bridge` every time because the inner CLI's own `# Environment`
   block outranked the system prompt; the system block + first-turn reminder
   added in this change invert that.
2. Ask the same question on a Mac. Expected:
       macOS (Sonoma / Sequoia / ...), working in /Users/<user>/<repo>
   If a single SwitchYard instance is reachable from both machines, the
   two answers come from the same sidecar but different sessions, and the
   host-symlink in plan name (or the per-request resolution) is what
   distinguishes them.

### 8c. What the offline suite cannot prove

The offline test suite (`for t in tests/test_*.py`) covers the parsers,
the probe-id round-trip and the synthetic probe response shape -- but it
cannot prove that LiteLLM's proxy actually forwards
`metadata.switchyard.caller_env` to the sidecar over HTTP. That transport
is exercised only by 8a above; if you change how the gateway writes
metadata or how the sidecar reads it, 8a is the verification step. The
sidecar is written so that field is optional: passive re-resolution is
the primary source, the stamp is belt-and-braces, and the system never
fails because the stamp is missing.



## Not verified here, and what to watch

These are the parts built from documentation rather than a live call, so check
them first if something misbehaves:

1. **Probe allowances.** Both probe endpoints are now verified against live
   sessions (host, path, field names and units), so this is no longer an open
   question — but MiniMax publishes *only* percentages, so its `allowance:` stays
   null and pacing has no token figure to aim at for those plans. The OpenCode Go
   limits ($12 / $30 / $60) came from its own payload.
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
