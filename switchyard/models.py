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

import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import yaml

CONFIG_PATH = os.environ.get("SWITCHYARD_PLANS", "/app/config/plans.yaml")
SEED_CAP = 2   # starting guess when a plan says `max_parallel: auto`

# Auth modes where a CLI harness stands between us and the model. That still
# matters for prompt-stacking and workspace isolation (see Plan.is_cli_backed),
# but it no longer implies anything about tool support — see Plan.can_use_tools.
CLI_AUTH = ("cli_sidecar", "oauth_sidecar")


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

    @property
    def ref(self) -> str:
        """How lanes and logs name it: `plan/model`."""
        return f"{self.plan_key}/{self.key}"

    @property
    def deployment(self) -> str:
        """The LiteLLM model_name generated for this pairing."""
        return f"sy.{self.plan_key}.{self.key}"

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
class Settings:
    drain_within_days: int = 21
    lease_ttl_seconds: int = 1800
    # Move inline <think>...</think> out of streamed content and into
    # reasoning_content, the way a buffered response already does.
    split_reasoning_tags: bool = True
    inflight_max_age_seconds: int = 120
    heartbeat_seconds: int = 30
    default_cooldown_seconds: int = 900
    concurrency_learning: ConcurrencyLearning = field(default_factory=ConcurrencyLearning)
    pacing: Pacing = field(default_factory=Pacing)


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

    def cap_for(self, model: Model) -> int:
        """A model may narrow the plan's limit, never widen it."""
        if model.max_parallel is None:
            return self.max_parallel
        return min(self.max_parallel, model.max_parallel)

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
class Lane:
    key: str
    label: str
    order: list[str]                 # `plan/model` refs
    tail: list[str] = field(default_factory=list)
    description: str = ""


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

    def siblings(self, model: Model) -> list[Model]:
        """Other live models on the same plan — they share its every limit."""
        return [m for m in self.plan_of(model).live_models if m.key != model.key]

    # -- lanes -------------------------------------------------------------
    def lane_members(self, lane_key: str) -> list[Model]:
        """Effective routing order for a lane.

        Two rules on top of the configured order: dead members drop out (plan
        disabled or expired, or the model disabled), and a member whose plan
        expires within `drain_within_days` is promoted ahead of members on plans
        that are not expiring, soonest death first, so cancelled capacity gets
        used before it vanishes. Tail members always stay last.
        """
        lane = self.lanes[lane_key]
        window = self.settings.drain_within_days

        def live(refs: list[str]) -> list[Model]:
            out: list[Model] = []
            for ref in refs:
                model = self.model(ref)
                if model is None:
                    continue
                plan = self.plans[model.plan_key]
                if plan.enabled and not plan.expired and model.enabled:
                    out.append(model)
            return out

        body = live(lane.order)

        def urgency(model: Model) -> tuple[int, int]:
            days = self.plans[model.plan_key].days_left
            return (0, days) if (days is not None and days <= window) else (1, 0)

        body.sort(key=urgency)
        return body + live(lane.tail)

    def is_tail(self, lane_key: str, ref: str) -> bool:
        return ref in self.lanes[lane_key].tail

    def lanes_using(self, model: Model) -> list[str]:
        return [k for k, lane in self.lanes.items()
                if model.ref in lane.order or model.ref in lane.tail]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
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
    return Probe(fields=fields, windows=windows, **body)


def _parse_models(plan_key: str, raw: Any) -> dict[str, Model]:
    if not raw:
        raise ValueError(f"plan {plan_key!r} declares no models")
    out: dict[str, Model] = {}
    for key, body in raw.items():
        body = dict(body or {})
        if "model" not in body:
            raise ValueError(f"model {plan_key}/{key} has no `model:` string")
        out[key] = Model(
            key=key,
            plan_key=plan_key,
            model=body["model"],
            label=body.get("label", ""),
            enabled=bool(body.get("enabled", True)),
            max_parallel=(int(body["max_parallel"]) if body.get("max_parallel") else None),
            context_window=(int(body["context_window"]) if body.get("context_window") else None),
        )
    return out


def load(path: str | None = None) -> Registry:
    with open(path or CONFIG_PATH) as fh:
        raw = yaml.safe_load(fh)

    sraw = dict(raw.get("settings") or {})
    settings = Settings(
        **{k: v for k, v in sraw.items() if k not in ("concurrency_learning", "pacing")},
        concurrency_learning=ConcurrencyLearning(**(sraw.get("concurrency_learning") or {})),
        pacing=Pacing(**(sraw.get("pacing") or {})),
    )

    plans: dict[str, Plan] = {}
    for key, body in (raw.get("plans") or {}).items():
        body = dict(body)
        quotas = _parse_quotas(body)
        probe = _parse_probe(body.pop("probe", None))
        models = _parse_models(key, body.pop("models", None))
        raw_parallel = body.get("max_parallel", 1)
        auto = str(raw_parallel).lower() == "auto"
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
            configured_parallel=None if auto else int(raw_parallel),
            max_parallel_ceiling=(int(body["max_parallel_ceiling"])
                                  if body.get("max_parallel_ceiling") else None),
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
    for key, body in (raw.get("lanes") or {}).items():
        lanes[key] = Lane(
            key=key,
            label=body.get("label", key),
            order=list(body.get("order") or []),
            tail=list(body.get("tail") or []),
            description=body.get("description", ""),
        )

    registry = Registry(settings=settings, plans=plans, lanes=lanes)

    # Fail loudly on a lane naming something that does not exist: a silent drop
    # would look like a routing bug much later.
    known = set(registry.models)
    for lane in lanes.values():
        for ref in list(lane.order) + list(lane.tail):
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
    return registry
