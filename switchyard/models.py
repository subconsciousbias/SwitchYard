"""Typed view over config/plans.yaml.

The structure mirrors what you actually buy. A **plan** owns the credentials,
the quota windows, the connection limit, the cost and the expiry date. A
**model** is something a plan can serve; it holds no credentials and may only
*narrow* its plan's connection limit, never widen it. Lanes reference models,
written `plan/model`.

Everything shared is shared because it belongs to the plan: two models on one
plan draw on the same slots, quota, cooldown and learned concurrency. There is
no separate grouping concept, because the plan already is one.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from typing import Any, Union

import yaml

CONFIG_PATH = os.environ.get("SWITCHYARD_PLANS", "/app/config/plans.yaml")
SEED_CAP = 2   # starting guess when a plan says `max_parallel: auto`

# Auth modes where a CLI harness stands between us and the model. That still
# matters for prompt-stacking and workspace isolation (see Plan.is_cli_backed),
# but it no longer implies anything about tool support — see Plan.can_use_tools.
CLI_AUTH = ("cli_sidecar", "oauth_sidecar")

# Default per-CLI tool blocklists. A request whose `x-switchyard-cli` header
# names a CLI here has those tools filtered out of its `tools` array before the
# picker ever sees the request — the harness's own file/shell tools reach the
# sidecar's container, not the caller's workspace, and silently doing the wrong
# thing there is worse than 400ing. The lists mirror what the sidecars already
# disable themselves; the gateway filter exists so the picker doesn't pin a
# mid-loop follow-up to a plan whose CLI cannot run the tools the caller's
# tools are, then 400 mid-turn. Matches are case-insensitive against the
# blocklist. Override per-deployment in settings.cli_tool_block.
#
#   opencode    — sidecars/cli_bridge/harness/opencode.json
#   claude-code — sidecars/cli_bridge/server.py (the claude harness)
#   codex       — no native tools to disable today
_DEFAULT_CLI_TOOL_BLOCK: dict[str, tuple[str, ...]] = {
    "opencode": (
        "bash", "edit", "write", "read", "grep", "glob", "list", "patch",
        "todowrite", "todoread", "webfetch", "websearch", "task", "multiedit",
    ),
    "claude-code": (
        "Bash", "Edit", "Write", "Read", "Glob", "Grep",
        "WebFetch", "WebSearch", "NotebookEdit",
    ),
    "codex": (),
}


@dataclass(frozen=True)
class Quota:
    """One quota window, belonging to a plan.

    Providers usually enforce several at once — a 5-hour burst window *and* a
    weekly allowance. Exactly one window per plan is the `target`: the allowance
    worth maximising, normally the weekly one. The others are `constraint`s: not
    to be overshot, but not goals in themselves.
    """
    kind: str = "unknown"          # tokens | dollars | window | unlimited | unknown
    period: str | None = None      # month | week | rolling_5h | day
    allowance: float | None = None
    source: str = "estimate"       # estimate | headers | probe | ledger | sidecar | none
    headers: dict[str, str] = field(default_factory=dict)
    name: str = ""
    role: str = "target"           # target | constraint

    @property
    def label(self) -> str:
        return self.name or (self.period or "window")

    @property
    def is_target(self) -> bool:
        return self.role == "target"


@dataclass(frozen=True)
class Probe:
    """How to read real headroom from a provider's own console endpoint.

    One response often carries every window the provider enforces at once — a
    weekly allowance *and* a 5-hour burst — so `windows` maps each quota window's
    name to its own field paths. `window` + `fields` remain the shorthand for a
    provider that publishes only one, and are folded into `windows` at load time
    so everything downstream reads one shape.
    """
    url: str
    kind: str = "cookie"              # cookie | bearer | none
    method: str = "GET"
    window: str | None = None
    interval_seconds: int = 600
    timeout_seconds: float = 15.0
    fields: dict[str, list[str]] = field(default_factory=dict)
    windows: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    # Multiplier applied to every number read. Providers report in their own
    # unit: OpenCode's Go meters are microcents, so 1e-8 turns them into dollars.
    scale: float = 1.0
    headers: dict[str, str] = field(default_factory=dict)
    referer: str = ""
    user_agent: str = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36")
    # Opt-in sliding capture of the session cookie. When True, a verified-good
    # probe response (2xx, no reauth marker, Set-Cookie present) overwrites the
    # stored credential in `sy:cred:{plan}`. Default False keeps today's exact
    # behaviour — capture only fires on plans that explicitly opt in and where
    # the console actually rotates the cookie on each successful request.
    capture_set_cookie: bool = False
    # Opt-in just-in-time login ceremony (issue #227). When True, an operator
    # can run `python3 -m switchyard.ceremony <plan>` on the host to open
    # `login_url` in a headful Chromium with an ephemeral profile, type
    # credentials into the real browser (they never reach SwitchYard), and on
    # reaching the post-auth console POST the harvested cookies to the portal's
    # existing `POST /admin/probes/{plan}/cookie` endpoint. Default False keeps
    # today's exact behaviour — the ceremony only fires on plans that explicitly
    # opt in, and the CLI itself refuses plans without the opt-in flag. The
    # login URL must share its host with `url` (validated at load time) so the
    # ceremony cannot exfiltrate cookies for a different site than the probe
    # already polls.
    login_ceremony: bool = False
    # Where `python3 -m switchyard.ceremony <plan>` opens the headful browser.
    # Required when `login_ceremony: true`; ignored otherwise. The host of this
    # URL must match the host of `url` — checked at load — so an operator
    # cannot point the ceremony at one provider and have it POST cookies to a
    # different plan's allowlist.
    login_url: str = ""


@dataclass(frozen=True)
class Model:
    """Something a plan serves. No credentials, no quota of its own."""
    key: str
    plan_key: str
    model: str                       # the provider string LiteLLM sends
    label: str = ""
    enabled: bool = True
    max_parallel: int | None = None   # narrows the plan's limit; never widens it
    context_window: int | None = None
    # Whether the model can serve image-bearing requests. None means "unset":
    # the loader pre-fills True for models litellm.model_cost marks as
    # vision-capable, and the picker treats None as False so a missing field
    # never grants vision. Operators can still set True or False explicitly
    # on any model to override the litellm-suggested default.
    supports_images: bool | None = None

    @property
    def ref(self) -> str:
        """How lanes and logs name it: `plan/model`."""
        return f"{self.plan_key}/{self.key}"

    @property
    def deployment(self) -> str:
        """The LiteLLM model_name generated for this pairing."""
        return f"sy.{self.plan_key}.{self.key}"

    @property
    def router_id(self) -> str:
        """The deterministic id we stamp onto ``model_info.id`` for this
        pairing. LiteLLM 1.101.0 stamps the router with a sha256 hexdigest
        of (model_name, litellm_params) when ``model_info.id`` is absent
        (router.py:9223-9225); without an explicit id the value is opaque
        and unmatchable from the response object's ``_hidden_params
        ["model_id"]``. The ``.id`` suffix keeps it distinct from the
        deployment string (model_name) so it does not collide with any
        litellm-model-cost-map entry.
        """
        return f"sy.{self.plan_key}.{self.key}.id"

    @property
    def display(self) -> str:
        return self.label or self.key


@dataclass(frozen=True)
class ConcurrencyLearning:
    enabled: bool = True
    buckets: str = "hour_of_day"
    seed_cap: int = SEED_CAP
    min_cap: int = 1
    decrease_factor: float = 0.5
    increase_step: int = 1
    min_samples: int = 3
    probe_interval_seconds: int = 300
    probe_cooldown_seconds: int = 900
    probe_pressure: int = 3


@dataclass(frozen=True)
class Pacing:
    enabled: bool = False
    min_slots: int = 1
    overshoot: float = 0.05
    disable_tail: bool = True
    include_metered: bool = False


@dataclass(frozen=True)
class TransientBreaker:
    """Per-plan circuit-breaker ladder for transient (5xx/timeout) failures.

    A plan that has just produced one 5xx still cools for the configured base
    (60s by default — see classify.Outcome.TRANSIENT); on the second in a row
    the ladder doubles the sit-out, then doubles again on the third, capped
    at `max_seconds` so a multi-hour outage cannot park the plan for the rest
    of the day. The streak is reset by the first successful call against the
    plan, so recovery is automatic. The board surfaces a 'failing · Nx' chip
    once the streak reaches `streak_alert`, which is the operator's cue that
    the re-pick-and-cool loop is doing its job but the plan itself needs a
    look.
    """
    enabled: bool = True
    max_seconds: int = 1800
    streak_alert: int = 3


@dataclass(frozen=True)
class CallerEnvironmentSettings:
    """How to resolve the caller's tool-execution environment per request.

    The caller's tools execute on the caller's machine, but the inner CLI
    runs in a SwitchYard relay container. Those two environments are not
    the same, and the model needs to be told so explicitly — otherwise
    the inner CLI's `# Environment` block (Linux, `/app/mcp_bridge`)
    outranks anything earlier in the system prompt and the model answers
    "I am on Linux in /app/mcp_bridge" to a Windows user (issue #44).

    Resolution precedence (see switchyard/caller_env.py for the full
    contract):

      1. config forced `platform` / `cwd` / `shell` -> source=config
      2. environment parsed from the request body    -> source=request
      3. synthetic probe (mcp_bridge tool round-trip)-> source=probe
      4. `fallback_platform` (PLATFORM ONLY)        -> source=host
      5. otherwise                                   -> source=unknown

    The relay's environment is never substituted as a fallback: the
    sidecar is the sidecar, and saying otherwise is the bug.

    `probe` controls whether the mcp_bridge may emit a synthetic
    environment-discovery tool call when neither (1) nor (2) gave an
    answer. `auto` is the right default for a permissive caller; a
    kiosk with no callable shell tool would still leave (4) and (5).
    `disabled` skips the probe entirely; `required` refuses requests
    that have neither an explicit override nor a passive environment.
    """
    probe: str = "auto"             # auto | disabled | required
    platform: str | None = None     # forced caller platform
    cwd: str | None = None          # forced caller working directory
    shell: str | None = None        # forced caller shell
    fallback_platform: str | None = None


@dataclass(frozen=True)
class Settings:
    drain_within_days: int = 21
    lease_ttl_seconds: int = 1800
    # How long a mid-tool-loop follow-up waits for a slot on its pinned plan
    # before spilling to a peer. Zero means spill immediately — the pin never
    # blocks. The wait keeps a loop on the plan holding the provider's prompt
    # cache during the short bursts when every slot is momentarily busy;
    # waiting longer than roughly one turn (10-20s) just moves the caller's
    # own retry delay into the gateway. Cooled plans and cap==0 plans are
    # never waited on.
    pin_wait_seconds: float = 10.0
    # How long an UNPINNED turn with an existing session lease waits for a
    # slot on its leased plan before spilling to a peer. Zero means spill
    # immediately — current behaviour, preserved as the default. The same
    # prompt-cache and burst-absorption case as `pin_wait_seconds` applies
    # to non-pinned turns that follow a session's previous pick, so giving
    # the lease a chance to land its plan before spilling reduces the
    # rate at which a chatty session bounces across plans. Cooled plans
    # and cap==0 plans are never waited on; the shared `ctx.wait` path
    # already enforces that for both wait knobs.
    affinity_wait_seconds: float = 0.0
    # Move inline <think>...</think> out of streamed content and into
    # reasoning_content, the way a buffered response already does.
    split_reasoning_tags: bool = True
    # Show the per-model $/session cell on the board (issue #76). Gates
    # rendering only -- the underlying HLL is always written so the number
    # is available the moment this is re-enabled.
    show_cost_per_session: bool = True
    inflight_max_age_seconds: int = 120
    heartbeat_seconds: int = 30
    default_cooldown_seconds: int = 900
    # Slots the gateway keeps free for CLI-backed plans, so the gateway's slot
    # table and the sidecar's own gate stop racing on the same number. The
    # physical gate (the sidecar reads `max_parallel` from this same file) stays
    # at the configured value; the gateway's effective cap is reduced by this
    # many slots, so an ordinary claim/release race or a release the gateway
    # makes (client abort) while the sidecar CLI turn is still finishing no
    # longer produces a spurious 429. API plans are unaffected: their provider
    # rejects on its own, the learner backs off, and the headroom would just
    # sit unused. Floor at 1 in the application so a single-connection plan
    # never drops to 0.
    gate_headroom_slots: int = 1
    # Per-CLI native-tool blocklist: a request whose `x-switchyard-cli` header
    # names a CLI here has those tool names filtered out of its `tools` array
    # before the picker runs, case-insensitive. The harness's own file/shell
    # tools reach the sidecar's container rather than the caller's workspace,
    # so a successful "edit" call there silently does the wrong thing — worse
    # than failing loudly. Mirrors what the sidecars already disable on their
    # side; see _DEFAULT_CLI_TOOL_BLOCK above for the source of each list.
    cli_tool_block: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {k: tuple(v) for k, v in _DEFAULT_CLI_TOOL_BLOCK.items()}
    )
    # Whether the picker should refuse to land an oversized request on a peer
    # whose `context_window` is too small. Default off (issue #104) preserves
    # today's behaviour: the LiteLLM-side `context_window_fallbacks` already
    # redirects an overflow to a larger-context peer, so non-opted operators
    # see no change. Setting this true moves the gate into the picker itself,
    # which means the overflow never reaches the provider's 4xx — useful for
    # plans that have no usable fallback chain (a single-member lane, or a
    # peer that just 400s on overflow rather than auto-failing over).
    enforce_context_window: bool = False
    concurrency_learning: ConcurrencyLearning = field(default_factory=ConcurrencyLearning)
    pacing: Pacing = field(default_factory=Pacing)
    transient_breaker: TransientBreaker = field(default_factory=TransientBreaker)
    caller_environment: "CallerEnvironmentSettings" = field(default_factory=CallerEnvironmentSettings)


@dataclass(frozen=True)
class Plan:
    """A thing you pay for. Owns credentials, quota, connection limit, expiry."""
    key: str
    label: str
    models: dict[str, Model]
    auth: str = "api_key"
    provider_family: str | None = None
    monthly_cost: float = 0.0
    metered: bool = False
    enabled: bool = True
    expires: date | None = None
    api_base: str | None = None
    api_key: str | None = None
    configured_parallel: int | None = None   # None when `max_parallel: auto`
    max_parallel_ceiling: int | None = None
    pacing: bool | None = None
    # None means "follow the global setting". Set `learning: false` on a plan
    # whose concurrency must stay exactly as configured.
    learning: bool | None = None
    # What to do once the provider reports the target window fully spent.
    # False (the default) stops routing to the plan until the window resets;
    # True keeps using it, which is right where a plan overflows into credits
    # or on-demand billing and you would rather spend that than queue.
    use_extra_quota: bool = False
    # How many tool-calling sessions may sit parked awaiting a caller's result.
    # A parked session runs no inference and holds no concurrency slot, but it
    # is a live CLI process, so it gets a limit of its own. 0 means "twice
    # max_parallel", which is the default.
    max_parked_sessions: int = 0
    supports_tools: bool | None = None
    quotas: tuple[Quota, ...] = field(default_factory=lambda: (Quota(),))
    probe: Probe | None = None
    alert_burn_rate_per_hour: float | None = None
    notes: str = ""

    # -- capacity ----------------------------------------------------------
    @property
    def max_parallel(self) -> int:
        """The plan's connection limit, or the seed when it is being learned."""
        return self.configured_parallel if self.configured_parallel is not None else SEED_CAP

    def cap_for(self, model: Model, settings: Settings | None = None) -> int:
        """A model may narrow the plan's limit, never widen it.

        For CLI-backed plans, the gateway-side effective cap is one slot of
        headroom below the plan's configured limit, so the sidecar's own gate
        (which reads the same `max_parallel` from this file) has room to absorb
        a claim/release race or a gateway release the sidecar CLI turn has not
        yet noticed. API plans keep the full configured cap — their provider
        enforces the limit, and a free headroom slot would just sit unused.
        Settings is passed in for the headroom value; the property stays a
        plain method so callers without a Registry (notably the LiteLLM
        config generator) can supply their own.
        """
        cap = self.max_parallel
        if model.max_parallel is not None:
            cap = min(cap, model.max_parallel)
        if settings is not None and self.is_cli_backed:
            cap = min(cap, max(1, self.max_parallel - settings.gate_headroom_slots))
        return cap

    # -- lifecycle ---------------------------------------------------------
    @property
    def expired(self) -> bool:
        return self.expires is not None and self.expires < date.today()

    @property
    def days_left(self) -> int | None:
        if self.expires is None:
            return None
        return (self.expires - date.today()).days

    # -- character ---------------------------------------------------------
    @property
    def can_use_tools(self) -> bool:
        """Whether a request carrying `tools` may be routed to this plan.

        Tool capability is a property of the plan, not an inference from how it
        is reached: it defaults to true, and a plan genuinely unable to serve a
        caller's tool definitions sets `supports_tools: false` in config to say
        so explicitly. (Historically every CLI-backed plan needed that override,
        because shelling out to an agent harness gave the caller's tools nowhere
        to go. That is being fixed per-provider — direct API access for some,
        an MCP bridge that hands the harness's tool call back to the caller for
        others — so the restriction is no longer assumed here.)
        """
        if self.supports_tools is not None:
            return self.supports_tools
        return True

    @property
    def is_cli_backed(self) -> bool:
        """Reached by shelling out to a vendor CLI rather than calling an API
        directly. Still meaningful for prompt-stacking and workspace isolation
        (see README), just not for tool capability any more."""
        return self.auth in CLI_AUTH

    @property
    def is_subscription(self) -> bool:
        """A fixed-fee plan with a resetting allowance — the only kind worth
        pacing. Metered providers and unlimited local models are not."""
        return not self.metered and self.quota.kind not in ("unlimited", "unknown")

    def paced(self, settings: Settings) -> bool:
        if self.pacing is not None:
            return bool(self.pacing) and settings.pacing.enabled
        if not settings.pacing.enabled:
            return False
        return self.is_subscription or (self.metered and settings.pacing.include_metered)

    @property
    def parked_limit(self) -> int:
        """Parked sessions allowed. Defaults to twice the connection limit."""
        return (self.max_parked_sessions if self.max_parked_sessions > 0
                else self.max_parallel * 2)

    def learns(self, settings: Settings) -> bool:
        """Whether the concurrency learner may move this plan's cap at all.

        Opt out per plan with `learning: false`. Worth doing wherever "no refusal"
        is not evidence of headroom: a local server queues requests instead of
        rejecting them, so the learner sees nothing but success and keeps
        probing, while the real effect is a growing queue and worse latency.
        """
        if not settings.concurrency_learning.enabled:
            return False
        return True if self.learning is None else bool(self.learning)

    # -- quota -------------------------------------------------------------
    @property
    def quota(self) -> Quota:
        """The window pacing aims to fill — the weekly allowance, usually."""
        for q in self.quotas:
            if q.is_target:
                return q
        return self.quotas[0] if self.quotas else Quota()

    @property
    def constraints(self) -> tuple[Quota, ...]:
        return tuple(q for q in self.quotas if not q.is_target)

    # -- models ------------------------------------------------------------
    @property
    def live_models(self) -> list[Model]:
        if not self.enabled or self.expired:
            return []
        return [m for m in self.models.values() if m.enabled]


@dataclass(frozen=True)
class Group:
    """A balancing strategy applied to a list of refs or nested groups.

    Groups are how a lane says "rotate through these", "weight these", "drain
    the emptiest of these", or "drain the one whose quota expires soonest".
    Members may themselves be `Group`s (recursive; depth limit ≤ 4) so a lane
    can rotate across inner `lowest_utilization` clusters, etc. The `gid` is a
    stable id derived from (lane, strategy, sorted-leaf-refs) so the picker
    can read and write a per-group Redis key without anyone having to mint one
    in config.
    """
    strategy: str                       # round_robin | weighted | lowest_utilization | perishable
    members: list[Any]                  # recursive: list[Ref | Group]
    weights: dict[str, int] | None = None   # weighted only: ref -> weight
    gid: str = ""


# A `Node` is one entry of `Lane.order`: either a bare `plan/model` ref string,
# or a `Group`. Recursive shape; not exposed as a separate dataclass so the
# type checker can keep treating the union as a flat thing.
Node = Union[str, Group]


_VALID_STRATEGIES = ("fill", "perishable")
_VALID_GROUP_STRATEGIES = ("round_robin", "weighted",
                           "lowest_utilization", "perishable")
_MAX_GROUP_DEPTH = 4


@dataclass(frozen=True)
class Lane:
    key: str
    label: str
    # Each entry is a `plan/model` ref (a bare string) or a `Group`. Flat
    # configs (refs only) parse unchanged so callers that only know about refs
    # keep working bit-for-bit; groups are added when the operator wants a
    # balancing strategy inside the lane. Tail stays a flat list of refs at
    # lane level — see the picker for how it is treated as the final stage.
    order: list[Any]
    tail: list[str] = field(default_factory=list)
    description: str = ""
    # "fill" (default) routes in config order, pure spill-and-fill. "perishable"
    # is sugar: when no explicit group is present, the picker wraps the order
    # in an implicit `{perishable: order}` group at the pick boundary. With
    # explicit groups in `order`, the lane-level strategy is ignored — the
    # groups already say how to order their members.
    strategy: str = "fill"
    # How to handle an image-bearing request on this lane.
    #
    #   off (default): image-bearing requests behave bit-for-bit like any
    #     other request. No filtering, no affinity changes — the lane's own
    #     strategy is the only thing that decides where a request lands.
    #   always: the picker filters members whose `supports_images` is not
    #     true, and an existing session lease on a non-image-capable model
    #     is dropped so the next image turn lands on a member that can
    #     serve it. When every image-capable member is at capacity or
    #     cooled, the request gets the usual LaneSaturated 429 — the lane
    #     never widens to text-only models to satisfy an image request.
    #   only_first: image-bearing requests still see the full member set
    #     (no member filtering), and the lane's existing affinity rules
    #     stay unchanged. The flag exists so a future caller can opt into
    #     "image-only first-time session" routing without breaking
    #     follow-ups; today it is the same as off.
    #
    # In every mode the lane's strategy (round_robin / weighted / perishable
    # / lowest_utilization / fill) is untouched — this only narrows the
    # candidate set and adjusts the affinity check.
    image_routing: str = "off"


@dataclass
class Registry:
    settings: Settings
    plans: dict[str, Plan]
    lanes: dict[str, Lane]

    # -- lookups -----------------------------------------------------------
    @property
    def models(self) -> dict[str, Model]:
        return {m.ref: m for p in self.plans.values() for m in p.models.values()}

    def model(self, ref: str) -> Model | None:
        plan_key, _, model_key = ref.partition("/")
        plan = self.plans.get(plan_key)
        return plan.models.get(model_key) if plan else None

    def plan_of(self, model: Model) -> Plan:
        return self.plans[model.plan_key]

    def model_for_deployment(self, deployment: str) -> Model | None:
        for m in self.models.values():
            if m.deployment == deployment:
                return m
        return None

    def model_for_router_id(self, router_id: str) -> Model | None:
        """Resolve a router-side deployment id (``_hidden_params["model_id"]``)
        back to a known Model. LiteLLM 1.101.0 fills this with the deployment's
        ``model_info.id`` at router init (router.py:8676), which
        ``switchyard/gen_litellm.py`` now stamps with ``Model.router_id`` --
        a deterministic string ``sy.{plan}.{model}.id``. Without the stamp
        the router fills the id with a sha256 hexdigest of (model_name,
        litellm_params) and there is no way to map it back. ``model_for_deployment``
        is for callers that pass a deployment string (the picker, the
        pre-call hook's direct-pick path); this is for the post-call side,
        where LiteLLM hands us the router id and we need to recognise it.
        """
        for m in self.models.values():
            if m.router_id == router_id:
                return m
        return None

    def siblings(self, model: Model) -> list[Model]:
        """Other live models on the same plan — they share its every limit."""
        return [m for m in self.plan_of(model).live_models if m.key != model.key]

    # -- lanes -------------------------------------------------------------
    def lane_nodes(self) -> dict[str, list[Any]]:
        """Parsed `Lane.order` for every lane: each entry is a `Ref` (bare
        `plan/model` string) or a `Group`. Tail is NOT included — the picker
        treats tail as the final stage on its own, not part of the body tree.

        This is the tree the picker walks; everything else that needs to
        enumerate a lane's refs (`routing_order`, `lane_members`,
        `_recompute_perishable_for_plan`) reads through this.
        """
        return {key: list(lane.order) for key, lane in self.lanes.items()}

    def routing_order(self, lane_key: str) -> list[str]:
        """Flat list of refs in the lane's declared order.

        Groups are flattened — `round_robin: [a, b]` walks a, b; `weighted:
        {a: 5, b: 2}` walks a, b in insertion order; nested groups recurse
        depth-first. A single-fill group around the body would also flatten
        bit-for-bit, but flat configs produce the same flat output as before:
        the lane-level `strategy: fill` is just config order, and a bare
        ref-list `order` is exactly the legacy shape.
        """
        out: list[str] = []
        for node in self.lanes[lane_key].order:
            self._flatten(node, out)
        return out

    @staticmethod
    def _flatten(node: Any, out: list[str]) -> None:
        if isinstance(node, Group):
            if node.weights:
                out.extend(node.weights.keys())
            else:
                for m in node.members:
                    Registry._flatten(m, out)
            return
        out.append(node)

    def lane_members(self, lane_key: str) -> list[Model]:
        """Effective routing order for a lane.

        The configured order with dead members dropped (plan disabled or
        expired, or the model disabled). Tail members always stay last.

        Expiry does NOT reorder anything here. Whether a cancelled plan should
        jump the queue depends on live quota state -- is it in its last window,
        and is that window behind pace? -- so the drain rule lives in the
        picker (`Picker._drain_first`), which can read the ledger.

        Group structure is collapsed before filtering: a `{round_robin: [a,
        b]}` and the flat `[a, b]` produce the same effective ordering, and a
        `{weighted: {a: 5, b: 2}}` walks a then b in declaration order — the
        weighting is the picker's job, not the body's.
        """
        refs = self.routing_order(lane_key)

        def live(ref_list: list[str]) -> list[Model]:
            out: list[Model] = []
            for ref in ref_list:
                model = self.model(ref)
                if model is None:
                    continue
                plan = self.plans[model.plan_key]
                if plan.enabled and not plan.expired and model.enabled:
                    out.append(model)
            return out

        return live(refs) + live(self.lanes[lane_key].tail)

    def is_tail(self, lane_key: str, ref: str) -> bool:
        return ref in self.lanes[lane_key].tail

    def lanes_using(self, model: Model) -> list[str]:
        """Every lane that names `model`, in body or tail. Group flattening is
        implicit — `routing_order` walks the tree and `is_tail` checks the
        lane-level tail list, so a `Ref` nested inside a `Group` is found."""
        return [k for k, lane in self.lanes.items()
                if model.ref in self.routing_order(k) or model.ref in lane.tail]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def router_signature(reg: Registry) -> str:
    """Fingerprint of everything the generated LiteLLM config bakes in.

    LiteLLM builds one deployment per plan/model pairing at startup, carrying
    the model string, api_base, credentials and context windows. Only a change
    to that surface forces `docker compose restart gateway` — a deployment the
    router has never seen cannot be routed to, whatever the hook says.

    Everything else — caps, quota windows, lane order, settings, pacing, costs,
    expiry — is read from the registry at request time, so those edits hot-swap
    without a restart. `max_parallel` is excluded on purpose: it lands in the
    generated config only as a backstop (`max_parallel_requests`), while the
    real gate is the slot table, which reads the live registry.
    """
    parts = sorted(
        (m.plan_key, m.key, m.model, reg.plan_of(m).api_base,
         reg.plan_of(m).api_key, m.context_window)
        for m in reg.models.values()
    )
    lanes = tuple(sorted(reg.lanes))
    blob = repr((parts, lanes)).encode()
    return hashlib.sha256(blob).hexdigest()


def _parse_date(v: Any) -> date | None:
    if v in (None, ""):
        return None
    if isinstance(v, date):
        return v
    return datetime.strptime(str(v), "%Y-%m-%d").date()


def _parse_quotas(body: dict) -> tuple[Quota, ...]:
    """`quotas:` (a list) wins; `quota:` (a single mapping) still works.

    With several windows and none marked, the longest period becomes the target:
    filling the weekly allowance is nearly always the goal, and the 5-hour window
    nearly always the thing not to overshoot.
    """
    raw = body.pop("quotas", None)
    if raw is None:
        single = dict(body.pop("quota", None) or {})
        return (Quota(**single),) if single else (Quota(),)

    order = {"rolling_5h": 0, "day": 1, "week": 2, "month": 3, None: 4}
    parsed = [Quota(**dict(entry)) for entry in raw]
    if not any(q.is_target for q in parsed):
        longest = max(parsed, key=lambda q: order.get(q.period, 4))
        parsed = [Quota(**{**q.__dict__, "role": "target" if q is longest else "constraint"})
                  for q in parsed]
    elif sum(1 for q in parsed if q.is_target) > 1:
        raise ValueError(f"only one quota window may be role: target, got {raw}")
    return tuple(parsed)


def _listify(mapping: Any) -> dict[str, list[str]]:
    return {k: (v if isinstance(v, list) else [v])
            for k, v in (mapping or {}).items()}


def _parse_cli_tool_block(raw: Any) -> dict[str, tuple[str, ...]]:
    """Coerce settings.cli_tool_block into a `cli -> tuple(tool)` mapping.

    Absent key falls back to _DEFAULT_CLI_TOOL_BLOCK; an outright malformed
    entry (non-mapping outer, non-list inner) raises — the field is operator-
    facing and a typo that drops the default to `{}` would silently let every
    CLI's native tools reach the wire. Loud-parse on purpose, matching the
    rest of `load()`.
    """
    if raw is None:
        return {k: tuple(v) for k, v in _DEFAULT_CLI_TOOL_BLOCK.items()}
    if not isinstance(raw, dict):
        raise ValueError(
            f"settings.cli_tool_block must be a mapping of cli -> tool list, "
            f"got {type(raw).__name__}: {raw!r}")
    out: dict[str, tuple[str, ...]] = {}
    for cli, names in raw.items():
        if not isinstance(cli, str):
            raise ValueError(
                f"settings.cli_tool_block keys must be CLI name strings, "
                f"got {type(cli).__name__}: {cli!r}")
        key = cli.strip().lower()
        if not key:
            raise ValueError(
                f"settings.cli_tool_block keys must be non-empty CLI name "
                f"strings, got whitespace-only: {cli!r}")
        if not isinstance(names, list):
            raise ValueError(
                f"settings.cli_tool_block[{cli!r}] must be a list of tool "
                f"name strings, got {type(names).__name__}: {names!r}")
        parsed: list[str] = []
        for name in names:
            if not isinstance(name, str):
                raise ValueError(
                    f"settings.cli_tool_block[{cli!r}] entries must be tool "
                    f"name strings, got {type(name).__name__}: {name!r}")
            parsed.append(name)
        out[key] = tuple(parsed)
    return out


def _parse_probe(raw: Any) -> Probe | None:
    if not raw:
        return None
    body = dict(raw)
    fields = _listify(body.pop("fields", None))
    windows = {name: _listify(paths)
               for name, paths in (body.pop("windows", None) or {}).items()}
    # The single-window shorthand is just one entry, so nothing downstream has
    # to know which form the config used.
    if fields and not windows:
        windows = {body.get("window") or "": fields}
    login_ceremony = bool(body.pop("login_ceremony", False))
    login_url = str(body.pop("login_url", "") or "")
    if login_ceremony:
        # `login_ceremony: true` without a target URL is the worst kind of
        # mistake — the CLI would refuse the plan at runtime, but a plan
        # whose probe silently broke because the operator forgot to fill
        # in the URL is worse. Loud-parse at load, matching the rest of
        # the probe checks (kind: cookie without `url`, etc.).
        if not login_url:
            raise ValueError(
                "probe.login_ceremony: true requires probe.login_url to be set")
        # Host pinning: the ceremony's allowlist is derived from `url`'s
        # host, so a `login_url` on a different host would silently let the
        # operator type credentials into one site and POST cookies for
        # another. Force the two URLs to share a host at load.
        url_host = _url_host(body.get("url", ""))
        login_host = _url_host(login_url)
        if not url_host or not login_host:
            raise ValueError(
                f"probe.login_ceremony requires both probe.url and "
                f"probe.login_url to be parseable URLs with hosts; got "
                f"url={body.get('url')!r}, login_url={login_url!r}")
        if url_host != login_host:
            raise ValueError(
                f"probe.login_ceremony requires login_url host to match "
                f"probe.url host (so the cookie allowlist matches the page "
                f"the operator is logging into); got url host {url_host!r} "
                f"and login_url host {login_host!r}")
    return Probe(fields=fields, windows=windows,
                 login_ceremony=login_ceremony, login_url=login_url, **body)


def _url_host(value: Any) -> str:
    """Host of an http(s) URL, or "" if the URL is unparseable.

    Used by the loader to pin `probe.login_url` to `probe.url`'s host when
    `probe.login_ceremony: true`. Local import: this function only fires
    when an operator sets `login_ceremony: true`, which is opt-in, so
    keeping `from urllib.parse import urlparse` out of the module's
    top-level import surface means a probe-less load path doesn't pay
    the cost. Returns the host lower-cased so a `URL` vs `url` casing
    difference in config does not split validation.
    """
    if not isinstance(value, str) or not value:
        return ""
    try:
        from urllib.parse import urlparse
        host = (urlparse(value).hostname or "").lower()
    except Exception:
        return ""
    return host


def _parse_model_max_parallel(plan_key: str, model_key: str, raw: Any) -> int | None:
    """Coerce and validate a model's `max_parallel`.

    Reject `0` and negative values rather than silently dropping them to
    None: a model that wants no cap on its plan is spelled `enabled: false`,
    not `max_parallel: 0` — the truthiness check the loader used to apply
    turned `0` into `None`, which let a plan keep its full configured cap
    while the operator expected the model to be throttled to zero.
    Non-int strings fail loudly with the offending model named, matching
    `_parse_cli_tool_block`'s convention. Floats are rejected outright
    rather than silently truncated by `int()`: a YAML `3.0` would land as
    `3` and `3.5` also as `3` with no warning, hiding a typo until the
    learned cap diverged from the configured one.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        # `bool` is a subclass of `int`, but "True" / "False" are not caps.
        raise ValueError(
            f"model {plan_key}/{model_key} max_parallel must be an integer "
            f">= 1, got {type(raw).__name__}: {raw!r}")
    if isinstance(raw, float):
        # `int()` silently truncates 3.0 -> 3 and -0.5 -> 0; the latter
        # then raises the misleading "got 0" below. Force a loud failure
        # so the operator sees the value they actually typed.
        raise ValueError(
            f"model {plan_key}/{model_key} max_parallel must be an integer "
            f">= 1, got {type(raw).__name__}: {raw!r}")
    try:
        n = int(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"model {plan_key}/{model_key} max_parallel must be an integer "
            f">= 1, got {type(raw).__name__}: {raw!r}") from None
    if n < 1:
        raise ValueError(
            f"model {plan_key}/{model_key} max_parallel must be an integer "
            f">= 1, got {n} (to disable a model, set enabled: false)")
    return n


def _parse_models(plan_key: str, raw: Any) -> dict[str, Model]:
    if not raw:
        raise ValueError(f"plan {plan_key!r} declares no models")
    out: dict[str, Model] = {}
    for key, body in raw.items():
        body = dict(body or {})
        if "model" not in body:
            raise ValueError(f"model {plan_key}/{key} has no `model:` string")
        supports_images_raw = body.get("supports_images")
        if supports_images_raw is None:
            supports_images: bool | None = None
        else:
            supports_images = bool(supports_images_raw)
        out[key] = Model(
            key=key,
            plan_key=plan_key,
            model=body["model"],
            label=body.get("label", ""),
            enabled=bool(body.get("enabled", True)),
            max_parallel=_parse_model_max_parallel(
                plan_key, key, body.get("max_parallel")),
            context_window=(int(body["context_window"]) if body.get("context_window") else None),
            supports_images=supports_images,
        )
    return out


# Modes accepted on a lane's `image_routing` key. Anything else is a loud
# parse error in `load()` — operators typing "all" or "true" want loud
# failure rather than silent no-op behaviour.
_IMAGE_ROUTING_MODES = ("always", "only_first", "off")


def litellm_known_vision_models() -> set[str]:
    """Model names litellm reports as vision-capable in `litellm.model_cost`.

    The lazy import + broad except is intentional: liteLLM may be absent (a
    fresh checkout running only the offline suite), the import path may
    shift across versions, or a malformed `model_cost` entry may blow up
    during iteration. None of those failures should stop `load()` from
    returning a Registry the rest of the gateway can use; they just mean
    the auto-fill of `supports_images = True` is unavailable for that run,
    and operators can still mark models explicitly in config.

    Each entry in `model_cost` is `{name: {supports_vision: True|False, ...}}`;
    we collect the names whose entry says `supports_vision` is True. The set
    is then used in `load()` to pre-fill `Model.supports_images` for any
    model whose key or `model:` string matches — but operator-set values
    (True or False in YAML) always win.
    """
    try:
        import litellm                                 # noqa: F401
        from litellm import model_cost
    except Exception:
        return set()
    out: set[str] = set()
    try:
        entries = model_cost.items()
    except Exception:
        return set()
    for name, info in entries:
        if not isinstance(info, dict):
            continue
        try:
            if bool(info.get("supports_vision")):
                out.add(str(name))
        except Exception:
            continue
    return out


def litellm_known_context_windows() -> dict[str, int]:
    """Model-name -> `max_input_tokens` from `litellm.model_cost`.

    The mirror of `litellm_known_vision_models`, but for context size: a
    pre-fill that lets `load()` know a model's window without the operator
    having to type it. The `Picker` enforces a per-peer overflow gate only
    when `settings.enforce_context_window` is true (default off, issue
    #104); the LiteLLM-side `context_window_fallbacks` is what handles the
    common case of an overflow that has a larger peer to redirect to, and
    it stays the load-bearing mechanism for non-opted operators.

    Each entry in `model_cost` is `{name: {max_input_tokens: N, ...}}`;
    only positive integers are kept — a missing, non-int or zero/negative
    value is treated as "no useful number", and the entry is skipped.
    Lazy import + broad except for the same reasons as
    `litellm_known_vision_models`: a missing or shifted litellm must not
    stop `load()` from returning a Registry. Returns `{}` on any failure.
    """
    try:
        import litellm                                 # noqa: F401
        from litellm import model_cost
    except Exception:
        return {}
    out: dict[str, int] = {}
    try:
        entries = model_cost.items()
    except Exception:
        return {}
    for name, info in entries:
        if not isinstance(info, dict):
            continue
        try:
            value = info.get("max_input_tokens")
        except Exception:
            continue
        if isinstance(value, bool):
            # bool is an int subclass; "True"/"False" are not windows.
            continue
        if not isinstance(value, int):
            continue
        if value <= 0:
            continue
        out[str(name)] = value
    return out


def _group_id(lane_key: str, strategy: str, leaf_refs: list[str]) -> str:
    """A stable id for a group, derived from lane + strategy + sorted leaves.

    Same config -> same id; a config edit that changes the leaves produces a
    different id, which is exactly the property the picker needs to write and
    read per-group Redis keys without anyone having to invent an id in YAML.
    """
    blob = f"{lane_key}|{strategy}|{','.join(sorted(leaf_refs))}".encode()
    return "g_" + hashlib.sha1(blob).hexdigest()[:12]


def _leaf_refs(node: Any) -> list[str]:
    """Refs reachable from `node`, recursively, in declaration order.

    Used both for gid computation and for ref-existence validation, so a ref
    buried inside a `{weighted: {a: 5, b: 2}}` inside a `{round_robin: [...]}`
    is still checked.
    """
    if isinstance(node, Group):
        if node.weights is not None:
            return list(node.weights.keys())
        out: list[str] = []
        for m in node.members:
            out.extend(_leaf_refs(m))
        return out
    return [node]


def _parse_lane_order(lane_key: str, raw: list, known: set[str],
                      depth: int = 1) -> list:
    """Parse the raw `order:` list into a list of `Ref` strings or `Group`s.

    Each entry is either a bare string (treated as a `plan/model` ref) or a
    mapping with exactly one key naming a group strategy
    (`round_robin | weighted | lowest_utilization | perishable`). The recursive
    depth limit (`_MAX_GROUP_DEPTH`) is checked here so a malformed YAML
    cannot make the picker walk forever.

    `known` may be empty during a partial parse; ref-existence is enforced
    there separately.
    """
    parsed: list = []
    for idx, entry in enumerate(raw):
        path = f"order[{idx}]"
        if isinstance(entry, str):
            if entry not in known and known:
                raise ValueError(
                    f"lane {lane_key!r} {path} references unknown model "
                    f"{entry!r}; known models: {sorted(known)}")
            parsed.append(entry)
            continue
        if not isinstance(entry, dict) or not entry:
            raise ValueError(
                f"lane {lane_key!r} {path} must be a 'plan/model' string or a "
                f"single-key strategy mapping, got {entry!r}")
        if len(entry) != 1:
            raise ValueError(
                f"lane {lane_key!r} {path} must have exactly one strategy key, "
                f"got {sorted(entry.keys())}")
        strategy, value = next(iter(entry.items()))
        if strategy not in _VALID_GROUP_STRATEGIES:
            raise ValueError(
                f"lane {lane_key!r} {path} uses unknown strategy {strategy!r}; "
                f"must be one of {_VALID_GROUP_STRATEGIES}")
        if depth > _MAX_GROUP_DEPTH:
            raise ValueError(
                f"lane {lane_key!r} {path} exceeds max group nesting depth "
                f"{_MAX_GROUP_DEPTH}")
        if strategy == "weighted":
            if not isinstance(value, dict) or not value:
                raise ValueError(
                    f"lane {lane_key!r} {path} weighted group must be a "
                    f"non-empty mapping of ref -> weight, got {value!r}")
            weights: dict[str, int] = {}
            for ref, w in value.items():
                if not isinstance(ref, str):
                    raise ValueError(
                        f"lane {lane_key!r} {path} weighted group key must be "
                        f"a 'plan/model' string, got {ref!r}")
                if not isinstance(w, int) or isinstance(w, bool):
                    raise ValueError(
                        f"lane {lane_key!r} {path} weight for {ref!r} must be "
                        f"a positive integer, got {w!r}")
                if w <= 0:
                    raise ValueError(
                        f"lane {lane_key!r} {path} weight for {ref!r} must be "
                        f"positive, got {w}")
                weights[ref] = w
            leaves = list(weights.keys())
            for ref in leaves:
                if known and ref not in known:
                    raise ValueError(
                        f"lane {lane_key!r} {path} weighted member references "
                        f"unknown model {ref!r}; known models: {sorted(known)}")
            parsed.append(Group(strategy=strategy, members=[],
                               weights=weights, gid=_group_id(lane_key, strategy, leaves)))
            continue
        # The other three strategies take a list of nodes (refs or nested groups).
        if not isinstance(value, list):
            raise ValueError(
                f"lane {lane_key!r} {path} {strategy!r} group must be a list, "
                f"got {value!r}")
        if not value:
            raise ValueError(
                f"lane {lane_key!r} {path} {strategy!r} group must not be empty")
        members = _parse_lane_order(lane_key, value, known, depth=depth + 1)
        leaves: list[str] = []
        for m in members:
            leaves.extend(_leaf_refs(m))
        parsed.append(Group(strategy=strategy, members=members,
                            weights=None, gid=_group_id(lane_key, strategy, leaves)))
    return parsed


def _parse_plan_max_parallel(plan_key: str, raw: Any) -> int | None:
    """Coerce and validate a plan's `max_parallel`.

    The plan's cap is either the literal string `auto` (case-insensitive),
    which becomes the learner-managed `None` once stored on the dataclass,
    or an integer >= 1. Anything else -- a non-int string, a negative
    integer, or zero -- is a load-time error: a negative cap would crash
    `asyncio.Semaphore(-N)`, zero would silently strand the plan forever,
    and "auto with a typo" (`"Auto"` works, `"auto-pilot"` does not) is the
    kind of thing a quiet truthiness parse used to swallow. The plan key
    is named in the message so an operator reading the failure knows
    exactly which plan to fix. Floats are rejected outright rather than
    silently truncated by `int()`: a YAML `3.0` would land as `3` and
    `3.5` also as `3`, hiding a typo until the next restart hit
    `asyncio.Semaphore(3)` instead of the cap the operator typed.
    """
    if isinstance(raw, bool):
        raise ValueError(
            f"plan {plan_key!r} max_parallel must be 'auto' or an integer "
            f">= 1, got {type(raw).__name__}: {raw!r}")
    if isinstance(raw, str):
        if raw.strip().lower() == "auto":
            return None
        raise ValueError(
            f"plan {plan_key!r} max_parallel must be 'auto' or an integer "
            f">= 1, got string: {raw!r}")
    if isinstance(raw, float):
        # `int()` silently truncates 3.0 -> 3 and -0.5 -> 0; the latter
        # then raises the misleading "got 0" below. Force a loud failure
        # so the operator sees the value they actually typed.
        raise ValueError(
            f"plan {plan_key!r} max_parallel must be 'auto' or an integer "
            f">= 1, got {type(raw).__name__}: {raw!r}")
    try:
        n = int(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"plan {plan_key!r} max_parallel must be 'auto' or an integer "
            f">= 1, got {type(raw).__name__}: {raw!r}") from None
    if n < 1:
        raise ValueError(
            f"plan {plan_key!r} max_parallel must be 'auto' or an integer "
            f">= 1, got {n} (to disable a plan, set enabled: false)")
    return n


def _parse_plan_max_parallel_ceiling(
        plan_key: str, raw: Any, configured_parallel: int | None) -> int | None:
    """Coerce and validate a plan's `max_parallel_ceiling`.

    The ceiling caps the learner -- a `max_parallel: auto` plan can probe
    upward up to this number. It must be an integer >= 1 when set, and
    when the plan also has a concrete `max_parallel` (i.e. not `auto`),
    the ceiling must be at least that number: a ceiling below the
    configured cap is a contradiction the learner cannot satisfy.

    None (the field omitted) is allowed and means "no ceiling". The
    truthiness check the loader used to apply -- `int(body[k]) if
    body.get(k) else None` -- silently dropped `0` to None and so could
    never catch a typo. Using `is not None` keeps the off-by-one between
    `0` and "unset" visible at load time. Floats are rejected outright
    rather than silently truncated by `int()`: a YAML `3.0` would land as
    `3` and `3.5` also as `3`, hiding a typo until the learned cap
    diverged from the configured one (or `asyncio.Semaphore(3)` opened
    fewer slots than the operator typed).
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise ValueError(
            f"plan {plan_key!r} max_parallel_ceiling must be an integer "
            f">= 1, got {type(raw).__name__}: {raw!r}")
    if isinstance(raw, float):
        # `int()` silently truncates 3.0 -> 3 and -0.5 -> 0; the latter
        # then raises the misleading "got 0" below. Force a loud failure
        # so the operator sees the value they actually typed.
        raise ValueError(
            f"plan {plan_key!r} max_parallel_ceiling must be an integer "
            f">= 1, got {type(raw).__name__}: {raw!r}")
    try:
        n = int(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"plan {plan_key!r} max_parallel_ceiling must be an integer "
            f">= 1, got {type(raw).__name__}: {raw!r}") from None
    if n < 1:
        raise ValueError(
            f"plan {plan_key!r} max_parallel_ceiling must be an integer "
            f">= 1, got {n}")
    if configured_parallel is not None and n < configured_parallel:
        raise ValueError(
            f"plan {plan_key!r} max_parallel_ceiling must be >= "
            f"max_parallel ({configured_parallel}), got {n}")
    return n


def load(path: str | None = None) -> Registry:
    with open(path or CONFIG_PATH) as fh:
        raw = yaml.safe_load(fh)

    sraw = dict(raw.get("settings") or {})
    cli_block = _parse_cli_tool_block(sraw.pop("cli_tool_block", None))
    settings = Settings(
        **{k: v for k, v in sraw.items() if k not in (
            "concurrency_learning", "pacing", "transient_breaker",
            "caller_environment")},
        cli_tool_block=cli_block,
        concurrency_learning=ConcurrencyLearning(**(sraw.get("concurrency_learning") or {})),
        pacing=Pacing(**(sraw.get("pacing") or {})),
        transient_breaker=TransientBreaker(**(sraw.get("transient_breaker") or {})),
        caller_environment=CallerEnvironmentSettings(
            **(sraw.get("caller_environment") or {})),
    )

    plans: dict[str, Plan] = {}
    for key, body in (raw.get("plans") or {}).items():
        body = dict(body)
        quotas = _parse_quotas(body)
        probe = _parse_probe(body.pop("probe", None))
        models = _parse_models(key, body.pop("models", None))
        configured_parallel = _parse_plan_max_parallel(
            key, body.get("max_parallel", 1))
        max_parallel_ceiling = _parse_plan_max_parallel_ceiling(
            key, body.get("max_parallel_ceiling"), configured_parallel)
        plans[key] = Plan(
            key=key,
            label=body.get("label", key),
            models=models,
            auth=body.get("auth", "api_key"),
            provider_family=body.get("provider_family"),
            monthly_cost=float(body.get("monthly_cost", 0) or 0),
            metered=bool(body.get("metered", False)),
            enabled=bool(body.get("enabled", True)),
            expires=_parse_date(body.get("expires")),
            api_base=body.get("api_base"),
            api_key=body.get("api_key"),
            configured_parallel=configured_parallel,
            max_parallel_ceiling=max_parallel_ceiling,
            pacing=body.get("pacing"),
            learning=body.get("learning"),
            use_extra_quota=bool(body.get("use_extra_quota", False)),
            max_parked_sessions=int(body.get("max_parked_sessions") or 0),
            supports_tools=body.get("supports_tools"),
            quotas=quotas,
            probe=probe,
            alert_burn_rate_per_hour=body.get("alert_burn_rate_per_hour"),
            notes=body.get("notes", ""),
        )

    lanes: dict[str, Lane] = {}
    # The set the parser checks refs against is derived from the parsed plans
    # so a `Group` nested in another `Group` still resolves — refs must point
    # at something that exists before the Registry object exists.
    known = {m.ref for p in plans.values() for m in p.models.values()}

    for key, body in (raw.get("lanes") or {}).items():
        strategy = str(body.get("strategy", "fill"))
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"lane {key!r} strategy must be one of {_VALID_STRATEGIES}, "
                f"got {strategy!r}")
        image_routing = str(body.get("image_routing", "off"))
        if image_routing not in _IMAGE_ROUTING_MODES:
            raise ValueError(
                f"lane {key!r} image_routing must be one of "
                f"{_IMAGE_ROUTING_MODES}, got {image_routing!r}")
        order = _parse_lane_order(key, list(body.get("order") or []), known)
        lanes[key] = Lane(
            key=key,
            label=body.get("label", key),
            order=order,
            tail=list(body.get("tail") or []),
            description=body.get("description", ""),
            strategy=strategy,
            image_routing=image_routing,
        )

    registry = Registry(settings=settings, plans=plans, lanes=lanes)

    # Tail is a flat list of refs at lane level — it is NOT walked by
    # `_parse_lane_order`, so check it here against the now-built registry.
    for lane in lanes.values():
        for ref in lane.tail:
            if ref not in known:
                raise ValueError(
                    f"lane {lane.key!r} references unknown model {ref!r}; "
                    f"known models: {sorted(known)}")

    # A tail must survive the moment it is needed. Its entire job is to keep a
    # lane alive once the paid capacity is exhausted, so a subscription in the
    # tail is self-defeating: that is precisely what will have run out. Local
    # models never run out, and a metered provider fails on money rather than
    # quota, so both are admissible; a subscription is not.
    for lane in lanes.values():
        for ref in lane.tail:
            plan = registry.plan_of(registry.models[ref])
            if plan.is_subscription and not plan.metered:
                raise ValueError(
                    f"lane {lane.key!r} has {ref!r} in its tail, but plan "
                    f"{plan.key!r} is a subscription. A tail exists for when the "
                    "paid capacity is gone, so it must be a local or metered "
                    "plan — otherwise it is exhausted exactly when needed.")

    # Pre-fill `supports_images = True` for any model the operator left unset
    # and whose key or `model:` string matches a litellm.model_cost entry
    # marked vision-capable. Operator values (True or False) are never
    # overwritten — litellm is a hint, the operator is the source of truth.
    # Models is a frozen dataclass, so a True pre-fill is a replacement;
    # only the affected plans/keys are rebuilt, not the whole registry.
    known_vision = litellm_known_vision_models()
    if known_vision:
        touched: dict[str, Plan] = {}
        for plan_key, plan in registry.plans.items():
            rebuilt: dict[str, Model] = {}
            for model_key, model in plan.models.items():
                if model.supports_images is not None:
                    continue
                if (model_key in known_vision
                        or model.model in known_vision):
                    rebuilt[model_key] = replace(model, supports_images=True)
            if rebuilt:
                merged = {**plan.models, **rebuilt}
                touched[plan_key] = replace(plan, models=merged)
        if touched:
            plans = {**registry.plans, **touched}
            registry = Registry(
                settings=registry.settings, plans=plans, lanes=registry.lanes,
            )

    # Pre-fill `context_window` from `litellm.model_cost` for any model the
    # operator left at None. Same contract as the supports_images pre-fill:
    # operator-set values win, the litellm hint only fills the unset gap.
    # The picker refuses to land an oversized request on a peer with a known
    # small window only when `settings.enforce_context_window` is true; this
    # pre-fill just makes the gate have data to act on without a config edit.
    known_context = litellm_known_context_windows()
    if known_context:
        touched_ctx: dict[str, Plan] = {}
        for plan_key, plan in registry.plans.items():
            rebuilt_ctx: dict[str, Model] = {}
            for model_key, model in plan.models.items():
                if model.context_window is not None:
                    continue
                cap = (known_context.get(model_key)
                       or known_context.get(model.model))
                if cap is None:
                    continue
                rebuilt_ctx[model_key] = replace(model, context_window=cap)
            if rebuilt_ctx:
                merged_ctx = {**plan.models, **rebuilt_ctx}
                touched_ctx[plan_key] = replace(plan, models=merged_ctx)
        if touched_ctx:
            plans = {**registry.plans, **touched_ctx}
            registry = Registry(
                settings=registry.settings, plans=plans, lanes=registry.lanes,
            )
    return registry
