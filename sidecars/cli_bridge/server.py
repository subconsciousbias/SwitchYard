"""OpenAI-compatible shim over a vendor's own CLI, for OAuth-only plans.

Why this exists: LiteLLM authenticates with static API keys, but a Claude Max
subscription and a ChatGPT seat are both OAuth-only, and their tokens live in
the respective CLI's credential store. So the CLI stays the client — it owns
login and refresh — and this shim just speaks HTTP on one side and the CLI on
the other. No OAuth token ever reaches LiteLLM.

One process per plan, selected with PROVIDER:
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
SwitchYard additionally refuses to route a request containing `tools` to any
CLI-backed plan, and this process rejects one outright rather than dropping the
definitions silently. Text in, text out is the contract.

One CLI can front several subscriptions — OpenCode logged into both xAI and
OpenCode Zen, for instance. Those are separate quotas, so each gets its own
process with its own SWITCHYARD_SUBSCRIPTION; sharing one process would conflate
two connection limits into one gate.

Concurrency, the default model and the allowed model aliases all come from
`config/plans.yaml` — the same file SwitchYard routes from — so there is exactly
one place to change a connection limit. SWITCHYARD_SUBSCRIPTION names which
subscription this process serves; every plan sharing it contributes its model
alias, and the tightest `max_parallel` among them is the connection limit.

The thing it must get exactly right is error mapping: a usage-limit rejection
has to leave here as **HTTP 429 with Retry-After**, because that is the signal
SwitchYard uses to drop the plan's slots out of the lane. A 500 would look like
a transient blip and the lane would keep feeding requests to dead capacity.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import yaml

log = logging.getLogger("cli_bridge")

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

app = FastAPI(title="switchyard-cli-bridge")

PROVIDER = os.environ.get("PROVIDER", "claude").lower()
# The plan this process fronts. SWITCHYARD_SUBSCRIPTION is the old name, kept so
# an existing compose file keeps working.
PLAN = (os.environ.get("SWITCHYARD_PLAN")
        or os.environ.get("SWITCHYARD_SUBSCRIPTION", ""))
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
        "model": os.environ.get("OPENCODE_MODEL", "xai/grok-4.6"),
        # --agent switchyard selects the minimal agent in harness/opencode.json:
        # every tool disabled and NO prompt field. Measured on a trivial call,
        # 7,239 -> 423 tokens, a 94% cut — the difference between a subscription
        # being usable for volume and not.
        #
        # The agent deliberately has no `prompt:`. Carrying a one-line prompt cost
        # 582 tokens against 423 without it, for identical answers, and it was
        # redundant anyway: the caller's system prompt is folded into the message,
        # so it governs. (`--pure` changes nothing; no plugins are installed.)
        "args": os.environ.get(
            "OPENCODE_ARGS",
            "run --model {model} --format json --agent switchyard {prompt}").split(),
        # No system-prompt FLAG at all, verified by testing: --prompt, --system
        # and --system-prompt each exit 1 as unknown options.
        #
        # An agent's `prompt:` field does work — it replaces the base prompt
        # rather than adding to it. An earlier note here said it had no effect,
        # on the strength of an agent told to ignore the user and answer a fixed
        # token; the model answered the user instead. That proved nothing: an
        # instruction to disregard the user is injection-shaped, and refusing it
        # is correct behaviour, not evidence the field was ignored. A neutral
        # marker in the same field was obeyed immediately.
        #
        # It is still omitted, for a different reason: no base prompt file
        # matches `xai/grok-*`, so there is nothing to replace and the field is
        # pure cost. The agent's `tools:` config is where the token cut comes
        # from. The caller's prompt is folded into the message; see fold_system.
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
        # --skip-git-repo-check: the sidecar's working directory is not a git
        #   repo, and codex otherwise refuses with "Not inside a trusted directory".
        # --model: the ids follow a scheme I guessed wrong at first — not
        #   gpt-5/gpt-5-codex but gpt-5.6-sol / -terra / -luna and gpt-6-astra.
        #   An unknown id is rejected with "The '<id>' model is not supported when
        #   using Codex with a ChatGPT account", so confirm before setting one.
        # model_instructions_file replaces the compiled-in base instructions.
        #   Measured: 14,255 -> 10,015 tokens on a trivial call. The residue is
        #   codex's own tool schema; include_plan_tool / include_apply_patch_tool /
        #   tools.web_search all had no effect, and experimental_instructions_file
        #   barely moved it (14,154), so this is the key that works.
        "args": os.environ.get(
            "CODEX_ARGS",
            "exec --json --skip-git-repo-check --model {model} {prompt}").split(),
        # Written per request from the caller's system prompt, which makes this a
        # real override rather than an append — the same contract as Claude's
        # --system-prompt.
        "instructions_arg": "-c model_instructions_file={path}",
        "instructions_default": "/app/harness/codex-instructions.md",
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
    models: set   # aliases a request may ask for
    source: str = "config"   # config | fallback — reported by /health


_config: Config | None = None
_config_at = 0.0


def read_config() -> Config:
    """Concurrency and model aliases for this plan, read from plans.yaml.

    The config nests models under their plan, so this reads
    `plans[<plan>].models` — a plan owns the connection limit, and each model
    contributes the alias this CLI will accept. An earlier version read a
    top-level `deployments:` map keyed by plan, which no longer exists; it found
    no models and silently fell back to the profile defaults, so every sidecar
    reported a model nobody had configured.

    Env vars still win, as an escape hatch when the config is unreadable.
    """
    fallback_model = os.environ.get(f"{PROVIDER.upper()}_MODEL") or PROFILE["model"]
    env_conc = os.environ.get("SIDECAR_CONCURRENCY")

    cap: int | None = None
    models: set[str] = set()
    default: str | None = None
    target = PLAN or PROVIDER

    try:
        with open(PLANS_PATH) as fh:
            raw = yaml.safe_load(fh) or {}
        seed = int(((raw.get("settings") or {}).get("concurrency_learning")
                    or {}).get("seed_cap", 2))
        plan = (raw.get("plans") or {}).get(target) or {}
        if not plan:
            log.warning("no plan %r in %s; falling back to env", target, PLANS_PATH)
        else:
            raw_cap = plan.get("max_parallel", 1)
            cap = seed if str(raw_cap).lower() == "auto" else int(raw_cap)
            for key, body in (plan.get("models") or {}).items():
                body = body or {}
                if body.get("enabled") is False:
                    continue
                spec = body.get("model")
                if not spec:
                    continue
                # Strip the LiteLLM provider prefix: `openai/claude-opus-5` is
                # `claude-opus-5` to the CLI, and `openai/opencode-go/glm-5.3-flash`
                # keeps its provider/model shape.
                alias = spec.split("/", 1)[1] if "/" in spec else spec
                models.add(alias)
                if default is None:
                    default = alias
    except (OSError, ValueError, TypeError) as exc:
        log.warning("could not read %s (%s); falling back to env", PLANS_PATH, exc)

    concurrency = int(env_conc) if env_conc else (cap if cap else 1)
    model = default or fallback_model
    source = "config" if models else "fallback"
    if source == "fallback":
        # Loud, because this is how a sidecar ends up serving a model nobody
        # configured: it reads no models, quietly uses the profile default, and
        # looks healthy while doing it.
        log.error("read no models for plan %r from %s — falling back to %r. "
                  "Check SWITCHYARD_PLAN and the plan's `models:` block.",
                  target, PLANS_PATH, model)
    return Config(concurrency=max(1, concurrency), model=model,
                  models=models | {model}, source=source)


def config() -> Config:
    global _config, _config_at
    if _config is None or (time.time() - _config_at) > CONFIG_TTL:
        _config = read_config()
        _config_at = time.time()
    return _config


class Gate:
    """A concurrency gate whose limit can change between requests.

    A *new* request never queues: a full gate answers immediately so SwitchYard
    can spill to the next plan in the lane instead of holding a worker open.
    The one exception is acquire_waiting, used only to resume a tool-calling
    session that was already committed to this plan — see mcp_bridge.
    """

    def __init__(self) -> None:
        self._in_flight = 0
        self._lock = asyncio.Lock()
        self._freed = asyncio.Condition()

    @property
    def in_flight(self) -> int:
        return self._in_flight

    async def acquire(self, limit: int) -> bool:
        async with self._lock:
            if self._in_flight >= limit:
                return False
            self._in_flight += 1
            return True

    async def acquire_waiting(self, limit: int, timeout: float) -> bool:
        """Acquire, waiting up to `timeout` for a slot to come free.

        Callers are woken in arrival order, so a queue of resuming sessions is
        served first-come-first-served rather than by luck of scheduling.
        """
        deadline = time.monotonic() + timeout
        if await self.acquire(limit):
            return True
        async with self._freed:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                try:
                    await asyncio.wait_for(self._freed.wait(), remaining)
                except asyncio.TimeoutError:
                    return False
                if await self.acquire(limit):
                    return True

    async def release(self) -> None:
        async with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
        async with self._freed:
            self._freed.notify()          # hand the slot to the longest waiter


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
def normalise(text: str) -> str:
    """Fold typographic punctuation before matching.

    Codex writes "You\u2019ve hit your usage limit" with a curly apostrophe, so a
    pattern containing a straight quote silently fails to match — and an exhausted
    subscription then looks like a transient error, which is the one mistake that
    makes the router hammer a dead plan.
    """
    return (text or "").replace("\u2019", "'").replace("\u2018", "'") \
                       .replace("\u201c", '"').replace("\u201d", '"')


# Deliberately loose. Vendors reword these constantly, and the cost of a miss is
# asymmetric: a missed limit means a 502 that the router treats as transient and
# retries, while a false positive merely rests a healthy plan for a while.
_LIMIT = re.compile(
    r"(usage limit"                       # "hit your usage limit", "usage limit reached"
    r"|(hit|reached|exceeded) your"       # "you've hit your ...", "reached your ..."
    r"|limit (will )?reset"
    r"|purchase more credits"
    r"|out of (credits|usage|quota)"
    r"|insufficient (balance|credit|quota)"
    r"|rate.?limit"
    r"|too many requests"
    r"|quota)", re.I,
)
_RESET_AT = re.compile(r"reset(?:s|ting)?\s+at\s+([0-9]{1,2}(?::[0-9]{2})?\s*(?:am|pm)?)", re.I)
# Codex states an absolute reset: "try again at Sep 22nd, 2026 4:37 AM".
_TRY_AGAIN_AT = re.compile(
    r"try again at\s+([A-Z][a-z]{2,9}\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}"
    r"(?:\s+\d{1,2}:\d{2}(?:\s*[AaPp][Mm])?)?)", re.I)


def seconds_until(text: str) -> int | None:
    """Seconds until an absolute reset time stated in a CLI's error message.

    Worth the effort: without it a 31-hour lockout gets the default one-hour
    cooldown, so the lane retries a dead plan thirty more times. The stated time
    carries no timezone, so it is read as UTC; a result in the past or absurdly
    far out is discarded rather than trusted.
    """
    m = _TRY_AGAIN_AT.search(normalise(text))
    if not m:
        return None
    stamp = re.sub(r"(\d)(st|nd|rd|th)", r"\1", m.group(1), flags=re.I).replace(".", "")
    stamp = re.sub(r"\s+", " ", stamp).strip().rstrip(",")
    for fmt in ("%b %d, %Y %I:%M %p", "%b %d %Y %I:%M %p", "%B %d, %Y %I:%M %p",
                "%B %d %Y %I:%M %p", "%b %d, %Y %H:%M", "%b %d, %Y", "%B %d, %Y"):
        try:
            when = datetime.strptime(stamp, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        delta = (when - datetime.now(timezone.utc)).total_seconds()
        if 60 <= delta <= 14 * 86400:
            return int(delta)
        return None
    return None
_AUTH = re.compile(r"(not logged in|please run .?claude login|authentication|invalid credentials)", re.I)


def flatten(messages: list[dict]) -> tuple[str, str | None]:
    """Collapse a chat array into one prompt plus a system prompt.

    A limitation worth knowing: this is stateless, so each turn re-sends the
    whole conversation and pays for it. SwitchYard's session affinity keeps a
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


def _int(v) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def iter_json_objects(blob: str):
    """Yield every JSON object in a stream, whether JSONL or concatenated.

    OpenCode emits one object per line, but pretty-printed or run-together output
    would silently yield nothing under a line-based parser — and "no text found"
    then falls back to returning the raw stream as the answer, which is exactly
    the bug this replaces.
    """
    decoder = json.JSONDecoder()
    index = 0
    length = len(blob)
    while index < length:
        brace = blob.find("{", index)
        if brace < 0:
            return
        try:
            obj, end = decoder.raw_decode(blob, brace)
        except ValueError:
            index = brace + 1
            continue
        index = end
        if isinstance(obj, dict):
            yield obj


def parse_output(stdout: str, kind: str | None = None) -> dict:
    """Normalise a CLI's output into {result, usage}.

    `kind` defaults to this process's own PROFILE, but mcp_bridge calls this
    with its own provider's parser kind explicitly — the event shapes
    ("claude_json", "events_json", "codex_jsonl") are the same regardless of
    which sidecar is asking, so there is no reason to duplicate this parser.
    """
    kind = kind or PROFILE["parser"]

    if kind == "claude_json":
        return json.loads(stdout)

    if kind in ("events_json", "codex_jsonl"):
        # Two different event shapes, both verified against real output.
        #
        # OpenCode nests everything under `part`:
        #   {"type":"text","part":{"type":"text","text":"OK"}}
        #   {"type":"step_finish","part":{"type":"step-finish",
        #      "tokens":{"input":6194,"output":18,"reasoning":0,
        #                "cache":{"read":1280}}}}
        #
        # Codex uses `item` plus a top-level usage object:
        #   {"type":"item.completed","item":{"type":"agent_message","text":"OK"}}
        #   {"type":"turn.completed","usage":{"input_tokens":14159,
        #      "cached_input_tokens":12160,"output_tokens":7,
        #      "reasoning_output_tokens":0}}
        #
        # Reasoning tokens are counted into output because they are billed, but
        # reasoning *text* is never part of the answer.
        text_parts: list[str] = []
        usage: dict = {}

        def add(field: str, value) -> None:
            usage[field] = usage.get(field, 0) + _int(value)

        for evt in iter_json_objects(stdout):
            part = evt.get("part") if isinstance(evt.get("part"), dict) else {}
            item = evt.get("item") if isinstance(evt.get("item"), dict) else {}

            # --- OpenCode ------------------------------------------------
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
            tokens = part.get("tokens") if isinstance(part.get("tokens"), dict) else None
            if tokens:
                cache = tokens.get("cache") if isinstance(tokens.get("cache"), dict) else {}
                add("input_tokens", tokens.get("input"))
                add("output_tokens", tokens.get("output"))
                add("output_tokens", tokens.get("reasoning"))
                add("cache_read_tokens", cache.get("read"))
            if isinstance(part.get("cost"), (int, float)):
                # The provider's notional API cost. Recorded for visibility only:
                # on a prepaid subscription the marginal cost of a request is zero.
                usage["provider_cost"] = usage.get("provider_cost", 0.0) + float(part["cost"])

            # --- Codex ---------------------------------------------------
            if item.get("type") in ("agent_message", "message") and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
            top = evt.get("usage") if isinstance(evt.get("usage"), dict) else None
            if top:
                add("input_tokens", top.get("input_tokens") or top.get("prompt_tokens"))
                add("output_tokens", top.get("output_tokens") or top.get("completion_tokens"))
                add("output_tokens", top.get("reasoning_output_tokens"))
                add("cache_read_tokens", top.get("cached_input_tokens"))

            # --- generic fallbacks for shapes neither of the above covers --
            if not part and not item:
                msg = evt.get("message") or evt.get("text") or evt.get("delta")
                if isinstance(msg, dict):
                    msg = msg.get("content") or msg.get("text")
                if isinstance(msg, list):
                    msg = "".join(str(b.get("text", "")) for b in msg
                                  if isinstance(b, dict))
                if isinstance(msg, str) and msg.strip() and evt.get("type") in (
                        None, "message", "assistant", "agent_message",
                        "response.output_text.delta"):
                    text_parts.append(msg)

        if not text_parts:
            raise json.JSONDecodeError("no assistant text in CLI output", stdout, 0)
        return {"result": "".join(text_parts), "usage": usage}

    return {"result": stdout.strip()}


def fold_max_tokens(system: str | None, max_tokens: int | None) -> str | None:
    """Turn the caller's max_tokens into an instruction, since no CLI has a flag.

    `claude -p` and `opencode run` expose turn limits, not token caps, so a
    caller asking for 50 tokens would otherwise get a full-length answer — and
    pay the subscription quota for it. An instruction is a soft limit the model
    can ignore, but it genuinely shortens output, which is the point. /health
    reports enforces_max_tokens: false so this is not mistaken for a hard cap.
    """
    if not max_tokens or max_tokens <= 0:
        return system
    words = max(10, int(max_tokens * 0.7))
    hint = (f"Answer in at most roughly {words} words. Be direct: no preamble, "
            "no restating the question.")
    return f"{system}\n\n{hint}" if system else hint


@contextlib.contextmanager
def instructions_file(system: str | None):
    """Yield the argv fragment that overrides this CLI's base instructions.

    Codex takes its system prompt as a *file path* in config rather than a flag,
    so the caller's prompt is written to a temp file per request. That makes it a
    real replacement of the built-in instructions — the same contract as Claude's
    --system-prompt — instead of yet another layer stacked on top.

    Yields ([], system) for CLIs with no such mechanism, leaving the caller's
    text to be folded into the prompt instead.
    """
    template = PROFILE.get("instructions_arg")
    if not template:
        yield [], system
        return

    path = PROFILE.get("instructions_default")
    tmp = None
    if system:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False)
        tmp.write(system if system.endswith("\n") else system + "\n")
        tmp.close()
        path = tmp.name
    try:
        # The template is one token like `-c key={path}`; split so the value is
        # passed as a single argv element even when the path contains spaces.
        parts = [p.replace("{path}", path) for p in template.split(" ")]
        yield parts, None
    finally:
        if tmp:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


def fold_system(prompt: str, system: str | None) -> tuple[str, str | None]:
    """When a CLI has no system-prompt flag, put the caller's instructions at
    the top of the prompt rather than discarding them silently."""
    if PROFILE.get("instructions_arg"):
        # Handled by instructions_file(), which overrides rather than appends.
        return prompt, system
    key = ("system_args_replace" if SYSTEM_MODE == "replace"
           and PROFILE.get("system_args_replace") else "system_args")
    if system and not PROFILE.get(key):
        return f"{system}\n\n{prompt}", None
    return prompt, system


async def run_cli(prompt: str, system: str | None, model: str | None = None) -> dict:
    prompt, system = fold_system(prompt, system)
    with instructions_file(system) as (extra_args, system):
        return await _run_cli(prompt, system, model, extra_args)


async def _run_cli(prompt: str, system: str | None, model: str | None,
                   extra_args: list) -> dict:
    cmd = build_argv(prompt, system, model) + list(extra_args)

    # stdin must be closed explicitly: codex reads "additional input from stdin"
    # and would block forever on an inherited descriptor that never closes.
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=TIMEOUT)
    except asyncio.TimeoutError:
        proc.kill()
        raise HTTPException(status_code=408, detail=f"{PROVIDER} cli timed out")

    stdout, stderr = out.decode(errors="replace"), err.decode(errors="replace")
    blob = normalise(f"{stdout}\n{stderr}")

    if proc.returncode != 0 or not stdout.strip():
        if _AUTH.search(blob):
            raise HTTPException(status_code=401,
                                detail=f"{PROVIDER} cli not authenticated: {stderr[:300]}")
        up_status, up_message, up_retryable = upstream_error(stdout)

        # A quota message wins over the upstream status, even a non-retryable 403.
        # xAI answers an exhausted SuperGrok subscription with
        # "personal-team-blocked:spending-limit: You have run out of credits or
        # need a Grok subscription" / 403 / isRetryable:false — which reads
        # terminal but is a plan that refills. Calling it a dead credential would
        # sideline a healthy subscription.
        if _LIMIT.search(blob) or seconds_until(blob):
            raise _limit_error(f"{up_message}\n{blob}" if up_message else blob)

        # Otherwise an explicit upstream client error is the truth: a bad model
        # id, a blocked account, a rejected credential. Surface it as-is so the
        # router does not cool a plan down over something waiting cannot fix.
        if up_status in (401, 402, 403) and up_retryable is not True:
            raise HTTPException(
                status_code=up_status,
                detail={"error": {"message": up_message or blob[:300],
                                  "type": "entitlement_or_credentials"}})
        status, detail = error_from_events(stdout)
        if status and 400 <= status < 500 and status != 429:
            # A client error is our fault, not the provider's — surface it as-is
            # so SwitchYard does not cool the plan down over a bad request.
            raise HTTPException(status_code=status,
                                detail={"error": {"message": detail or blob[:300],
                                                  "type": "upstream_client_error"}})
        raise HTTPException(
            status_code=502,
            detail=f"{PROVIDER} cli failed ({proc.returncode}): "
                   f"{detail or stderr[:300] or stdout[:300]}")

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


def upstream_error(stdout: str) -> tuple[int | None, str, bool | None]:
    """(status, message, retryable) from an OpenCode APIError envelope.

    OpenCode reports the provider's own answer:
      {"type":"error","error":{"name":"APIError","data":{
         "message":"personal-team-blocked:spending-limit: You have run out of
                    credits or need a Grok subscription...",
         "statusCode":403,"isRetryable":false}}}
    That is far better signal than pattern-matching the prose: 403 with
    isRetryable false means no amount of waiting will help, so it must not be
    dressed up as a usage limit that resets in an hour.
    """
    for evt in iter_json_objects(stdout):
        err = evt.get("error") if isinstance(evt.get("error"), dict) else {}
        data = err.get("data") if isinstance(err.get("data"), dict) else {}
        if not data:
            continue
        status = data.get("statusCode")
        retryable = data.get("isRetryable")
        message = data.get("message")
        if isinstance(status, int) or isinstance(message, str):
            return (status if isinstance(status, int) else None,
                    str(message or "")[:500],
                    retryable if isinstance(retryable, bool) else None)
    return None, "", None


def error_from_events(stdout: str) -> tuple[int | None, str]:
    """(upstream status, message) from a CLI's error events.

    Codex reports failures as `{"type":"error","message":"{...nested json...}"}`
    on **stdout**, leaving stderr with an unrelated informational line. Reading
    only stderr turned a clear "model is not supported" into an opaque 502.
    """
    status: int | None = None
    messages: list[str] = []
    for evt in iter_json_objects(stdout):
        item = evt.get("item") if isinstance(evt.get("item"), dict) else {}
        for candidate in (evt, item, evt.get("error") if isinstance(evt.get("error"), dict) else {}):
            if not isinstance(candidate, dict):
                continue
            if candidate.get("type") in ("error", "turn.failed") or candidate.get("message"):
                msg = candidate.get("message")
                if isinstance(msg, str) and msg.strip():
                    messages.append(msg.strip())
        # The message is often itself JSON carrying the HTTP status.
        for msg in list(messages):
            if msg.startswith("{"):
                try:
                    inner = json.loads(msg)
                except ValueError:
                    continue
                if isinstance(inner, dict):
                    if isinstance(inner.get("status"), int):
                        status = inner["status"]
                    err = inner.get("error") if isinstance(inner.get("error"), dict) else {}
                    if isinstance(err.get("message"), str):
                        messages.append(err["message"])
    unique = list(dict.fromkeys(m for m in messages if not m.startswith("{")))
    return status, " | ".join(unique)[:500]


def _limit_error(blob: str, default_retry_after: int | None = None) -> HTTPException:
    # Default to the plan's window; if the CLI names a reset time, trust it.
    # mcp_bridge passes its own provider profile's window explicitly, since it
    # may be running as a different PROVIDER than this module was imported
    # under (see the note on read_config for how the two stay independent).
    retry_after = default_retry_after if default_retry_after is not None else PROFILE["default_retry_after"]
    detail = f"{PROVIDER} usage limit reached"

    blob = normalise(blob)
    absolute = seconds_until(blob)
    if absolute:
        retry_after = absolute
        hours = absolute / 3600
        detail = f"{detail} (resets in {hours:.1f}h)"
    else:
        m = _RESET_AT.search(blob)
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


# --------------------------------------------------------------- plan usage ---
# Where the CLI writes its session transcripts. `claude -p "/usage"` runs the
# same slash command the TUI does and records the report it fetched, which is
# the only way to read plan headroom without calling Anthropic's API with the
# subscription's own token -- the thing a subscription's terms do not allow.
CLAUDE_PROJECTS = Path(os.environ.get(
    "CLAUDE_PROJECTS_DIR", str(Path.home() / ".claude" / "projects")))
USAGE_TIMEOUT = float(os.environ.get("USAGE_TIMEOUT_SECONDS", "120"))


def _find_usage_report(since: float) -> dict | None:
    """The newest `usageReport` written to a transcript after `since`.

    Transcripts are JSONL, one object per line, and the report is nested
    somewhere inside the record for the command that produced it — the exact
    depth has moved between versions, so it is searched for by key rather than
    by a fixed path.
    """
    newest: tuple[float, dict] | None = None
    if not CLAUDE_PROJECTS.is_dir():
        return None
    for path in CLAUDE_PROJECTS.rglob("*.jsonl"):
        try:
            if path.stat().st_mtime < since - 5:
                continue
            text = path.read_text(errors="replace")
        except OSError:
            continue
        if "usageReport" not in text:
            continue
        for line in text.splitlines():
            if "usageReport" not in line:
                continue
            try:
                found = _dig_key(json.loads(line), "usageReport")
            except ValueError:
                continue
            if isinstance(found, dict) and found.get("rate_limits"):
                stamp = path.stat().st_mtime
                if newest is None or stamp > newest[0]:
                    newest = (stamp, found)
    return newest[1] if newest else None


def _dig_key(node, key: str):
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for value in node.values():
            found = _dig_key(value, key)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _dig_key(value, key)
            if found is not None:
                return found
    return None


CODEX_SESSIONS = Path(os.environ.get(
    "CODEX_SESSIONS_DIR", str(Path.home() / ".codex" / "sessions")))


def _codex_rate_limits() -> dict | None:
    """The newest `rate_limits` block Codex wrote to a session rollout.

    Codex records it on ordinary calls, so unlike Claude nothing has to be run
    to produce it — the cost is that it is only as fresh as the last request
    this sidecar made. `resets_at` says which window it describes, so a stale
    reading is still interpretable rather than silently wrong.
    """
    if not CODEX_SESSIONS.is_dir():
        return None
    files = sorted((p for p in CODEX_SESSIONS.rglob("*.jsonl")),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files[:25]:
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        if "rate_limits" not in text:
            continue
        for line in reversed(text.splitlines()):
            if "rate_limits" not in line:
                continue
            try:
                found = _dig_key(json.loads(line), "rate_limits")
            except ValueError:
                continue
            if isinstance(found, dict) and found.get("primary"):
                return {"rate_limits": found,
                        "observed_at": path.stat().st_mtime}
    return None


async def usage_report() -> dict:
    """What the vendor's own client last reported about this plan's headroom.

    Deliberately never a direct call to the vendor's usage API — Anthropic's
    api/oauth/usage or ChatGPT's backend-api/codex/usage — because reaching
    those from here means using the subscription's own OAuth token, which is
    what its terms do not allow. The CLI is asked instead, or its own records
    are read.
    """
    if PROVIDER == "codex":
        report = _codex_rate_limits()
        if report is None:
            raise HTTPException(
                status_code=503,
                detail="no rate_limits recorded yet — codex writes them on a "
                       "real request, so send one through this plan first")
        return report
    if PROVIDER != "claude":
        raise HTTPException(status_code=501,
                            detail=f"no usage report implemented for {PROVIDER!r}")
    started = time.time()
    proc = await asyncio.create_subprocess_exec(
        PROFILE["cli"], "-p", "/usage",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        cwd="/tmp")
    try:
        _, err = await asyncio.wait_for(proc.communicate(), USAGE_TIMEOUT)
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        raise HTTPException(status_code=504,
                            detail=f"/usage did not finish within {USAGE_TIMEOUT:.0f}s")
    if proc.returncode != 0:
        raise HTTPException(
            status_code=502,
            detail=f"/usage exited {proc.returncode}: {err.decode(errors='replace')[:300]}")

    report = _find_usage_report(started)
    if report is None:
        # The command ran but wrote nothing we recognise — a version change in
        # the transcript shape is the likely cause, and saying so beats
        # returning an empty report that reads as "no usage".
        raise HTTPException(
            status_code=502,
            detail="/usage produced no usageReport in the CLI's transcripts; the "
                   "transcript shape may have changed in this CLI version")
    return report


@app.get("/usage")
async def usage() -> dict:
    return await usage_report()


@app.get("/health")
async def health() -> dict:
    cfg = config()
    return {"ok": cfg.source == "config", "provider": PROVIDER,
            "config_source": cfg.source, "supports_tools": False,
            "home": os.environ.get("HOME", ""), "warm": _warm.is_set(),
            # The CLIs have no token cap, so max_tokens becomes a prompt
            # instruction: a real reduction, but not a guarantee.
            "enforces_max_tokens": False,
            "system_mode": SYSTEM_MODE, "bare": BARE,
            "plan": PLAN or PROVIDER, "model": cfg.model,
            "models": sorted(cfg.models), "concurrency": cfg.concurrency,
            "in_flight": _gate.in_flight, "config": PLANS_PATH}


@app.get("/v1/models")
async def models() -> dict:
    return {"object": "list",
            "data": [{"id": m, "object": "model", "owned_by": f"switchyard-{PROVIDER}"}
                     for m in sorted(config().models)]}


async def _handle_chat(body: dict):
    """The text-only completion path, factored out of the route so mcp_bridge
    can call it directly for a request with no `tools` — same gate, same
    config, same error classification, not a re-implementation of any of it.
    That identity is what "no regression on the text path" means here: there
    is only one code path for it, whichever sidecar is asking.
    """
    # Refuse tool calls loudly. SwitchYard already routes these away from
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
    # No CLI accepts a token cap, so express it as an instruction instead of
    # dropping it on the floor.
    system = fold_max_tokens(system, body.get("max_tokens"))
    if not prompt:
        raise HTTPException(status_code=400, detail="no usable message content")
    model, warning = resolve_model(body.get("model"))
    if warning:
        # Loud, because silently running a weaker model than the lane asked for
        # would make an `apex` escalation quietly indistinguishable from `judge`.
        log.warning("%s", warning)

    limit = config().concurrency
    if not await _gate.acquire(limit):
        # Never queue: SwitchYard needs to hear "full" immediately so it can
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


@app.post("/v1/chat/completions")
async def chat(request: Request):
    return await _handle_chat(await request.json())
