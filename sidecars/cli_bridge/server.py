"""OpenAI-compatible shim over a vendor's own CLI, for OAuth-only plans.

Why this exists: LiteLLM authenticates with static API keys, but a Claude Max
subscription and a ChatGPT seat are both OAuth-only, and their tokens live in
the respective CLI's credential store. So the CLI stays the client — it owns
login and refresh — and this shim just speaks HTTP on one side and the CLI on
the other. No OAuth token ever reaches LiteLLM.

One process per subscription, selected with PROVIDER:
    PROVIDER=claude    -> `claude -p`       (Claude Max)
    PROVIDER=codex     -> `codex exec`      (ChatGPT seat)
    PROVIDER=opencode  -> `opencode run`    (SuperGrok via OpenCode; OpenCode Go)

PROMPT STACKING, and why these lanes refuse tool calls. Each of these CLIs is a
whole agent harness: it has its own system prompt, its own tools and its own
loop. Pointing another harness (OpenCode, Cursor, Claude Code) at a lane backed
by one of them stacks two agent prompts, and the caller's tool definitions have
nowhere to go — the inner harness's tools act on *this container*, not the
caller's workspace, and their results never reach the caller.

Two mitigations, neither of which makes these lanes agentic:
  * SYSTEM_MODE=replace passes the caller's system prompt with the CLI's
    override flag (`--system-prompt`) instead of appending to the built-in one,
    so only one agent prompt is in play. In that mode the Claude CLI also gets
    `--exclude-dynamic-system-prompt-sections`, which drops the working
    directory, git state and environment blurbs it would otherwise inject — noise
    the caller pays for on every request. Verify both flags exist on your CLI
    version (`claude --help | grep system-prompt`); a wrong flag is a hard error,
    which is why the code default stays `append`.
  * BARE=1 strips the inner harness's tools and caps it at one turn, which is
    as close to a plain completion as a CLI gets.
Switchyard additionally refuses to route a request containing `tools` to any
CLI-backed plan, and this process rejects one outright rather than dropping the
definitions silently. Text in, text out is the contract.

One CLI can front several subscriptions — OpenCode logged into both xAI and
OpenCode Zen, for instance. Those are separate quotas, so each gets its own
process with its own SWITCHYARD_SUBSCRIPTION; sharing one process would conflate
two connection limits into one gate.

Concurrency, the default model and the allowed model aliases all come from
`config/plans.yaml` — the same file Switchyard routes from — so there is exactly
one place to change a connection limit. SWITCHYARD_SUBSCRIPTION names which
subscription this process serves; every plan sharing it contributes its model
alias, and the tightest `max_parallel` among them is the connection limit.

The thing it must get exactly right is error mapping: a usage-limit rejection
has to leave here as **HTTP 429 with Retry-After**, because that is the signal
Switchyard uses to drop the plan's slots out of the lane. A 500 would look like
a transient blip and the lane would keep feeding requests to dead capacity.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass

import yaml

log = logging.getLogger("cli_bridge")

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

app = FastAPI(title="switchyard-cli-bridge")

PROVIDER = os.environ.get("PROVIDER", "claude").lower()
SUBSCRIPTION = os.environ.get("SWITCHYARD_SUBSCRIPTION", "")
PLANS_PATH = os.environ.get("SWITCHYARD_PLANS", "/app/config/plans.yaml")
CONFIG_TTL = 30.0          # re-read plans.yaml this often, so edits land live
TIMEOUT = int(os.environ.get("SIDECAR_TIMEOUT", "600"))
# Escape hatch for CLI flags that differ by version, e.g. "--full-auto".
EXTRA_ARGS = [a for a in os.environ.get("CLI_EXTRA_ARGS", "").split() if a]
# replace = override the CLI's own agent prompt; append = stack on top of it.
SYSTEM_MODE = os.environ.get("SYSTEM_MODE", "append").lower()
# Strip the inner harness's tools and cap it at a single turn.
BARE = os.environ.get("BARE", "1") not in ("0", "false", "no")

# Per-provider invocation. `prompt` and `system` are substituted; `system` is
# dropped entirely when the CLI has no equivalent flag.
PROFILES: dict[str, dict] = {
    "claude": {
        "cli": os.environ.get("CLAUDE_CLI", "claude"),
        "model": os.environ.get("CLAUDE_MODEL", "opus"),
        "args": ["-p", "{prompt}", "--output-format", "json", "--model", "{model}"],
        # SYSTEM_MODE=replace swaps these for --system-prompt, which overrides
        # the CLI's own agent prompt instead of stacking on top of it. See the
        # prompt-stacking note in the module docstring.
        "system_args": ["--append-system-prompt", "{system}"],
        "system_args_replace": ["--system-prompt", "{system}"],
        # Strip the inner harness's own tools: they would act on the sidecar's
        # container, not the caller's workspace, and the caller never sees them.
        "bare_args": ["--max-turns", "1", "--disallowed-tools",
                      "Bash,Edit,Write,Read,Glob,Grep,WebFetch,WebSearch,NotebookEdit"],
        # Only valid alongside --system-prompt. Drops the CLI's dynamically
        # injected sections (working directory, git state, environment), which
        # are pure noise when the caller supplies its own prompt — and which the
        # caller is paying for on every single request.
        "replace_extra_args": ["--exclude-dynamic-system-prompt-sections"],
        "parser": "claude_json",
        # Reset-window hint the CLI prints when the 5h limit is hit.
        "default_retry_after": 5 * 3600,
    },
    # OpenCode, logged in to the provider whose subscription this process
    # fronts. Headless is `opencode run`, models are named provider/model, and
    # `--format json` emits event objects. This is how a SuperGrok subscription
    # is reached without Grok Build (which needs SuperGrok Heavy specifically),
    # and it serves OpenCode Zen plans the same way.
    "opencode": {
        "cli": os.environ.get("OPENCODE_CLI", "opencode"),
        "model": os.environ.get("OPENCODE_MODEL", "xai/grok-4"),
        "args": os.environ.get(
            "OPENCODE_ARGS", "run --model {model} --format json {prompt}").split(),
        "system_args": [],
        "parser": "events_json",
        "default_retry_after": 3600,
    },
    # NOTE: the Codex CLI's flags move between releases. Verify against the
    # installed version with `codex exec --help`; override with CLI_EXTRA_ARGS
    # or CODEX_ARGS rather than editing this file.
    "codex": {
        "cli": os.environ.get("CODEX_CLI", "codex"),
        "model": os.environ.get("CODEX_MODEL", "gpt-5"),
        "args": os.environ.get("CODEX_ARGS", "exec --json --model {model} {prompt}").split(),
        "system_args": [],
        "parser": "codex_jsonl",
        "default_retry_after": 3600,
    },
}

if PROVIDER not in PROFILES:
    raise SystemExit(f"PROVIDER must be one of {sorted(PROFILES)}, got {PROVIDER!r}")

PROFILE = PROFILES[PROVIDER]
CLI = PROFILE["cli"]


# ---------------------------------------------------------------------------
# Configuration comes from plans.yaml, not from the compose file. Duplicating a
# connection limit in two places means one of them is wrong the moment you
# change the other.
# ---------------------------------------------------------------------------
@dataclass
class Config:
    concurrency: int
    model: str
    models: set  # aliases a request may ask for


_config: Config | None = None
_config_at = 0.0


def read_config() -> Config:
    """Concurrency and model aliases for this subscription, from plans.yaml.

    * concurrency = the tightest `max_parallel` among the plans sharing this
      subscription, since they all share one connection;
    * models = the deployment alias of each *enabled* plan on it, so flipping a
      plan's `enabled:` is all it takes to allow its tier here;
    * model = the alias of the plan named exactly like the subscription.

    Env vars still win, as an escape hatch when the config is unavailable.
    """
    fallback_model = os.environ.get(f"{PROVIDER.upper()}_MODEL") or PROFILE["model"]
    env_conc = os.environ.get("SIDECAR_CONCURRENCY")

    caps: list[int] = []
    models: set[str] = set()
    default: str | None = None

    try:
        with open(PLANS_PATH) as fh:
            raw = yaml.safe_load(fh) or {}
        plans = raw.get("plans") or {}
        deployments = raw.get("deployments") or {}
        seed = int(((raw.get("settings") or {}).get("concurrency_learning")
                    or {}).get("seed_cap", 2))
        target = SUBSCRIPTION or PROVIDER
        for key, body in plans.items():
            body = body or {}
            if (body.get("subscription") or key) != target:
                continue
            if body.get("enabled") is False:
                continue
            raw_cap = body.get("max_parallel", 1)
            caps.append(seed if str(raw_cap).lower() == "auto" else int(raw_cap))
            model = (deployments.get(key) or {}).get("model")
            if model:
                alias = model.split("/", 1)[1] if "/" in model else model
                models.add(alias)
                if key == target:
                    default = alias
    except (OSError, ValueError, TypeError) as exc:
        log.warning("could not read %s (%s); falling back to env", PLANS_PATH, exc)

    concurrency = int(env_conc) if env_conc else (min(caps) if caps else 1)
    model = default or (sorted(models)[0] if models else fallback_model)
    return Config(concurrency=max(1, concurrency), model=model,
                  models=models | {model})


def config() -> Config:
    global _config, _config_at
    if _config is None or (time.time() - _config_at) > CONFIG_TTL:
        _config = read_config()
        _config_at = time.time()
    return _config


class Gate:
    """A concurrency gate whose limit can change between requests.

    Never queues: a full gate answers immediately so Switchyard can spill to the
    next plan in the lane instead of holding a worker open.
    """

    def __init__(self) -> None:
        self._in_flight = 0
        self._lock = asyncio.Lock()

    @property
    def in_flight(self) -> int:
        return self._in_flight

    async def acquire(self, limit: int) -> bool:
        async with self._lock:
            if self._in_flight >= limit:
                return False
            self._in_flight += 1
            return True

    async def release(self) -> None:
        async with self._lock:
            self._in_flight = max(0, self._in_flight - 1)


_gate = Gate()

# Concurrency is N CLI subprocesses inside THIS one container, all reading the
# same credential directory — not N containers, so one login covers every
# concurrent run. But a cold start with a token due for refresh would have every
# subprocess racing to refresh and rewrite that shared credential file at once.
# So the first request runs alone; once one has succeeded, the token is fresh and
# the rest can proceed at full concurrency.
_warm = asyncio.Event()
_warmup_lock = asyncio.Lock()


async def invoke(prompt: str, system: str | None, model: str | None) -> dict:
    if _warm.is_set():
        return await run_cli(prompt, system, model)
    async with _warmup_lock:
        if _warm.is_set():                     # someone warmed it while we waited
            return await run_cli(prompt, system, model)
        payload = await run_cli(prompt, system, model)
        _warm.set()
        log.info("%s: first call succeeded, releasing full concurrency", PROVIDER)
        return payload

# The CLI reports exhaustion in prose; these are the shapes worth trusting.
_LIMIT = re.compile(
    r"(usage limit reached|limit will reset|you've reached your|rate.?limit"
    r"|out of (credits|usage)|quota)", re.I,
)
_RESET_AT = re.compile(r"reset(?:s|ting)?\s+at\s+([0-9]{1,2}(?::[0-9]{2})?\s*(?:am|pm)?)", re.I)
_AUTH = re.compile(r"(not logged in|please run .?claude login|authentication|invalid credentials)", re.I)


def flatten(messages: list[dict]) -> tuple[str, str | None]:
    """Collapse a chat array into one prompt plus a system prompt.

    A limitation worth knowing: this is stateless, so each turn re-sends the
    whole conversation and pays for it. Switchyard's session affinity keeps a
    session pinned here, which is what makes the CLI's prompt cache effective.
    """
    system: list[str] = []
    turns: list[str] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            content = "\n".join(
                str(b.get("text", "")) for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        content = (content or "").strip()
        if not content:
            continue
        role = m.get("role")
        if role == "system":
            system.append(content)
        elif role == "assistant":
            turns.append(f"Assistant: {content}")
        else:
            turns.append(f"Human: {content}")
    return "\n\n".join(turns), ("\n\n".join(system) or None)


def resolve_model(requested: str | None) -> tuple[str, str | None]:
    """(model to run, warning). An unlisted request falls back to the default.

    A caller cannot make the CLI run an arbitrary model string: the alias has to
    belong to an enabled plan on this subscription. Falling back is logged
    loudly, because an `apex` escalation quietly served by the `judge` model is
    the kind of bug nobody notices.
    """
    cfg = config()
    if not requested or requested == cfg.model:
        return cfg.model, None
    if requested in cfg.models:
        return requested, None
    return cfg.model, (f"model {requested!r} is not an enabled plan on "
                       f"subscription {SUBSCRIPTION or PROVIDER!r} "
                       f"({sorted(cfg.models)}); ran {cfg.model} instead")


def build_argv(prompt: str, system: str | None, model: str | None = None) -> list[str]:
    model = model or config().model

    def fill(tpl: str) -> str:
        return tpl.replace("{prompt}", prompt).replace("{model}", model).replace(
            "{system}", system or "")

    argv = [CLI] + [fill(a) for a in PROFILE["args"]]

    key = ("system_args_replace" if SYSTEM_MODE == "replace"
           and PROFILE.get("system_args_replace") else "system_args")
    if system and PROFILE.get(key):
        argv += [fill(a) for a in PROFILE[key]]

    if BARE and PROFILE.get("bare_args"):
        argv += [fill(a) for a in PROFILE["bare_args"]]

    if key == "system_args_replace" and system and PROFILE.get("replace_extra_args"):
        argv += [fill(a) for a in PROFILE["replace_extra_args"]]

    return argv + EXTRA_ARGS


def parse_output(stdout: str) -> dict:
    """Normalise a CLI's output into {result, usage}."""
    kind = PROFILE["parser"]

    if kind == "claude_json":
        return json.loads(stdout)

    if kind in ("events_json", "codex_jsonl"):
        # A stream of JSON events (or one JSON envelope): the assistant text is
        # the answer and a token-count event carries usage. Unknown event shapes
        # are ignored rather than fatal, so a CLI update degrades instead of
        # breaking. Handles both JSONL and a single pretty-printed object.
        text_parts: list[str] = []
        usage: dict = {}

        # A single JSON object (pretty-printed or not) rather than JSONL.
        stripped = stdout.strip()
        if stripped.startswith("{") and "\n{" not in stripped:
            try:
                doc = json.loads(stripped)
            except json.JSONDecodeError:
                doc = None
            if isinstance(doc, dict):
                if isinstance(doc.get("usage"), dict):
                    usage.update(doc["usage"])
                for key in ("result", "response", "output", "text", "content"):
                    val = doc.get(key)
                    if isinstance(val, str) and val.strip():
                        return {"result": val, "usage": usage}

        for line in stdout.splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                if key in evt:
                    usage[key] = evt[key]
            if isinstance(evt.get("usage"), dict):
                usage.update(evt["usage"])
            msg = evt.get("message") or evt.get("text") or evt.get("delta")
            if isinstance(msg, dict):
                msg = msg.get("content") or msg.get("text")
            if isinstance(msg, list):
                msg = "".join(
                    str(b.get("text", "")) for b in msg if isinstance(b, dict)
                )
            if isinstance(msg, str) and msg.strip():
                if evt.get("type") in (None, "message", "assistant", "item.completed",
                                       "agent_message", "text", "part.text",
                                       "response.output_text.delta"):
                    text_parts.append(msg)
        if not text_parts:
            raise json.JSONDecodeError("no assistant text in CLI output", stdout, 0)
        return {"result": text_parts[-1] if len(text_parts) == 1 else "".join(text_parts),
                "usage": usage}

    return {"result": stdout.strip()}


def fold_system(prompt: str, system: str | None) -> tuple[str, str | None]:
    """When a CLI has no system-prompt flag, put the caller's instructions at
    the top of the prompt rather than discarding them silently."""
    key = ("system_args_replace" if SYSTEM_MODE == "replace"
           and PROFILE.get("system_args_replace") else "system_args")
    if system and not PROFILE.get(key):
        return f"{system}\n\n{prompt}", None
    return prompt, system


async def run_cli(prompt: str, system: str | None, model: str | None = None) -> dict:
    prompt, system = fold_system(prompt, system)
    cmd = build_argv(prompt, system, model)

    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=TIMEOUT)
    except asyncio.TimeoutError:
        proc.kill()
        raise HTTPException(status_code=408, detail=f"{PROVIDER} cli timed out")

    stdout, stderr = out.decode(errors="replace"), err.decode(errors="replace")
    blob = f"{stdout}\n{stderr}"

    if proc.returncode != 0 or not stdout.strip():
        if _AUTH.search(blob):
            raise HTTPException(status_code=401,
                                detail=f"{PROVIDER} cli not authenticated: {stderr[:300]}")
        if _LIMIT.search(blob):
            raise _limit_error(blob)
        raise HTTPException(status_code=502, detail=f"{PROVIDER} cli failed ({proc.returncode}): {stderr[:300]}")

    try:
        payload = parse_output(stdout)
    except (json.JSONDecodeError, ValueError):
        payload = {"result": stdout.strip()}

    # The CLI can exit 0 while reporting a limit inside the JSON envelope.
    if payload.get("is_error") or payload.get("subtype") in ("error_max_turns", "error_during_execution"):
        text = json.dumps(payload)
        if _LIMIT.search(text):
            raise _limit_error(text)
        raise HTTPException(status_code=502, detail=f"{PROVIDER} cli error: {text[:300]}")
    if _LIMIT.search(str(payload.get("result", ""))) and not payload.get("usage"):
        raise _limit_error(str(payload.get("result")))

    return payload


def _limit_error(blob: str) -> HTTPException:
    # Default to the plan's window; if the CLI names a reset time, trust it.
    retry_after = PROFILE["default_retry_after"]
    m = _RESET_AT.search(blob)
    detail = f"{PROVIDER} usage limit reached"
    if m:
        detail = f"{detail} (resets at {m.group(1)})"
    return HTTPException(
        status_code=429,
        detail={"error": {"message": detail, "type": "usage_limit_reached"}},
        headers={"Retry-After": str(retry_after)},
    )


def to_openai(payload: dict, model: str) -> dict:
    usage = payload.get("usage") or {}
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": payload.get("result", "")},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": int(usage.get("input_tokens", 0) or 0),
            "completion_tokens": int(usage.get("output_tokens", 0) or 0),
            "total_tokens": int(usage.get("input_tokens", 0) or 0) + int(usage.get("output_tokens", 0) or 0),
        },
    }


@app.get("/health")
async def health() -> dict:
    cfg = config()
    return {"ok": True, "provider": PROVIDER, "supports_tools": False,
            "home": os.environ.get("HOME", ""), "warm": _warm.is_set(),
            "system_mode": SYSTEM_MODE, "bare": BARE,
            "subscription": SUBSCRIPTION or PROVIDER, "model": cfg.model,
            "models": sorted(cfg.models), "concurrency": cfg.concurrency,
            "in_flight": _gate.in_flight, "config": PLANS_PATH}


@app.get("/v1/models")
async def models() -> dict:
    return {"object": "list",
            "data": [{"id": m, "object": "model", "owned_by": f"switchyard-{PROVIDER}"}
                     for m in sorted(config().models)]}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()

    # Refuse tool calls loudly. Switchyard already routes these away from
    # CLI-backed plans; if one arrives anyway, dropping the definitions silently
    # would look like the model simply choosing not to call anything.
    if body.get("tools"):
        raise HTTPException(
            status_code=400,
            detail={"error": {
                "message": (f"{PROVIDER} is a CLI-backed plan and cannot serve tool "
                            "calls: the caller's tools would have nowhere to run. "
                            "Route tool-using requests to an API-keyed plan."),
                "type": "tools_unsupported"}})

    prompt, system = flatten(body.get("messages") or [])
    if not prompt:
        raise HTTPException(status_code=400, detail="no usable message content")
    model, warning = resolve_model(body.get("model"))
    if warning:
        # Loud, because silently running a weaker model than the lane asked for
        # would make an `apex` escalation quietly indistinguishable from `judge`.
        log.warning("%s", warning)

    limit = config().concurrency
    if not await _gate.acquire(limit):
        # Never queue: Switchyard needs to hear "full" immediately so it can
        # spill to the next plan in the lane instead of blocking a worker.
        raise HTTPException(status_code=429,
                            detail=f"sidecar at capacity ({limit})",
                            headers={"Retry-After": "5"})
    try:
        payload = await invoke(prompt, system, model)
    finally:
        await _gate.release()
    result = to_openai(payload, model)

    if not body.get("stream"):
        return result

    async def one_shot():
        chunk = {
            "id": result["id"], "object": "chat.completion.chunk",
            "created": result["created"], "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant",
                         "content": result["choices"][0]["message"]["content"]},
                         "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        done = {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": result["usage"]}
        yield f"data: {json.dumps(done)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(one_shot(), media_type="text/event-stream")
