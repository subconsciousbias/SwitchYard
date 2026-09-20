"""Typed view over config/plans.yaml."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import yaml

CONFIG_PATH = os.environ.get("SWITCHYARD_PLANS", "/app/config/plans.yaml")


@dataclass(frozen=True)
class Quota:
    """One quota window.

    Providers usually enforce several at once — a 5-hour burst window *and* a
    weekly allowance, sometimes a monthly one too. Exactly one window per plan
    is the `target`: the allowance worth maximising, normally the weekly one.
    The others are `constraint`s — we must not blow them, but landing them at
    100% is not a goal in itself.
    """
    kind: str = "unknown"          # tokens | dollars | window | unlimited | unknown
    period: str | None = None      # month | week | rolling_5h | day
    allowance: float | None = None
    source: str = "estimate"       # estimate | headers | probe | ledger | sidecar | none
    headers: dict[str, str] = field(default_factory=dict)
    name: str = ""                 # display name, defaults to the period
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

    Some consoles (MiniMax's `/coding_plan/remains`) only answer a browser
    session, so `kind: cookie` pairs with a cookie the portal stores for you.
    Field paths are lists of candidates because vendors rename things without
    notice; run the probe once and the portal shows the raw response to map.
    """
    url: str
    kind: str = "cookie"              # cookie | bearer | none
    method: str = "GET"
    window: str | None = None         # which quota window these numbers describe
    interval_seconds: int = 600
    timeout_seconds: float = 15.0
    fields: dict[str, list[str]] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    referer: str = ""
    user_agent: str = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36")


@dataclass(frozen=True)
class Plan:
    key: str
    label: str
    deployment: str
    max_parallel: int
    monthly_cost: float = 0.0
    auth: str = "api_key"
    provider_family: str | None = None   # drives vendor error-code mapping
    subscription_key: str | None = None      # shared-state group; see `subscription`
    configured_parallel: int | None = None   # None when `max_parallel: auto`
    max_parallel_ceiling: int | None = None  # hard upper bound for learning
    pacing: bool | None = None               # per-plan override of the global switch
    supports_tools: bool | None = None       # None -> inferred from `auth`
    context_window: int | None = None        # tokens; drives context fallbacks
    metered: bool = False
    enabled: bool = True
    expires: date | None = None
    quotas: tuple[Quota, ...] = field(default_factory=lambda: (Quota(),))
    probe: Probe | None = None
    alert_burn_rate_per_hour: float | None = None
    notes: str = ""

    @property
    def quota(self) -> Quota:
        """The window pacing aims to fill — the weekly allowance, usually."""
        for q in self.quotas:
            if q.is_target:
                return q
        return self.quotas[0] if self.quotas else Quota()

    @property
    def constraints(self) -> tuple[Quota, ...]:
        """Windows we must not overshoot, e.g. a 5-hour burst limit."""
        return tuple(q for q in self.quotas if not q.is_target)

    @property
    def expired(self) -> bool:
        return self.expires is not None and self.expires < date.today()

    @property
    def days_left(self) -> int | None:
        if self.expires is None:
            return None
        return (self.expires - date.today()).days

    @property
    def can_use_tools(self) -> bool:
        """Whether a request carrying `tools` may be routed here.

        A CLI-backed plan cannot serve one. The sidecar drives a whole agent
        harness — its own system prompt, its own tools, its own loop — so the
        caller's tool definitions have nowhere to go, its tool results would
        come from the sidecar's workspace rather than the caller's, and the two
        system prompts stack. Text in, text out is the honest contract for
        those; anything agentic belongs on an API-keyed plan.
        """
        if self.supports_tools is not None:
            return self.supports_tools
        # `oauth_sidecar` is the historical name; `cli_sidecar` is the accurate
        # one, since some of these credentials are API keys that only the CLI
        # knows how to use (OpenCode Zen, for one). Either means "a CLI harness
        # is in the way", which is what actually decides this.
        return self.auth not in ("oauth_sidecar", "cli_sidecar")

    @property
    def subscription(self) -> str:
        """The billable entity that owns the connection slots and the quota.

        Usually the plan itself. But one subscription can expose several model
        tiers — a Claude Max plan serving both a regular and a heavy model — and
        those tiers share *one* connection limit and *one* quota. Declaring
        `subscription: claude-max` on each makes them share slots, cooldowns,
        usage accounting, learned concurrency and pacing, so two lanes cannot
        between them open two connections against a one-connection plan.
        """
        return self.subscription_key or self.key

    @property
    def shares_subscription(self) -> bool:
        return self.subscription_key is not None and self.subscription_key != self.key

    @property
    def is_subscription(self) -> bool:
        """A fixed-fee plan with a resetting allowance — the only kind worth
        pacing. Metered providers and unlimited local models are not."""
        return not self.metered and self.quota.kind not in ("unlimited", "unknown")

    @property
    def known_allowances(self) -> int:
        return sum(1 for q in self.quotas if q.allowance)

    def paced(self, settings: "Settings") -> bool:
        if self.pacing is not None:
            return self.pacing and settings.pacing.enabled
        if not settings.pacing.enabled:
            return False
        return self.is_subscription or (self.metered and settings.pacing.include_metered)


@dataclass(frozen=True)
class Lane:
    key: str
    label: str
    order: list[str]
    tail: list[str] = field(default_factory=list)
    description: str = ""


SEED_CAP = 2   # starting guess when a plan says `max_parallel: auto`


@dataclass(frozen=True)
class ConcurrencyLearning:
    """Discover the parallelism a provider really tolerates (AIMD)."""
    enabled: bool = True
    buckets: str = "hour_of_day"     # hour_of_day | none
    seed_cap: int = SEED_CAP
    min_cap: int = 1
    decrease_factor: float = 0.5     # halve on a connection-limit rejection
    increase_step: int = 1           # creep up by one when demand is unmet
    min_samples: int = 3             # evidence needed to trust an hour bucket
    probe_interval_seconds: int = 300
    probe_cooldown_seconds: int = 900  # quiet time needed since last rejection
    probe_pressure: int = 3          # denied claims needed before probing up


@dataclass(frozen=True)
class Pacing:
    """Land each subscription at ~100% consumption exactly at its rollover."""
    enabled: bool = False            # the on/off switch
    min_slots: int = 1
    overshoot: float = 0.05          # aim 5% hot so we finish at 100%, not 95%
    disable_tail: bool = True        # no local spillover while pacing
    include_metered: bool = False    # metered providers keep fixed caps


@dataclass(frozen=True)
class Settings:
    drain_within_days: int = 21
    lease_ttl_seconds: int = 1800
    inflight_max_age_seconds: int = 900
    default_cooldown_seconds: int = 900
    concurrency_learning: ConcurrencyLearning = field(default_factory=ConcurrencyLearning)
    pacing: Pacing = field(default_factory=Pacing)


@dataclass
class Registry:
    settings: Settings
    plans: dict[str, Plan]
    lanes: dict[str, Lane]

    def subscription_of(self, plan_key: str) -> str:
        """Shared-state key for a plan named by string."""
        plan = self.plans.get(plan_key)
        return plan.subscription if plan else plan_key

    def siblings(self, plan: Plan) -> list[Plan]:
        """Other plans sharing this subscription's slots and quota."""
        return [p for p in self.plans.values()
                if p.subscription == plan.subscription and p.key != plan.key]

    def plan_for_deployment(self, deployment: str) -> Plan | None:
        for p in self.plans.values():
            if p.deployment == deployment:
                return p
        return None

    def lane_members(self, lane_key: str) -> list[Plan]:
        """Effective routing order for a lane.

        Two rules on top of the configured order:
          * expired and disabled plans drop out entirely;
          * a plan expiring within `drain_within_days` is promoted ahead of
            plans that are not expiring, so cancelled capacity gets used up.
        Tail plans always stay last, in their configured order.
        """
        lane = self.lanes[lane_key]
        window = self.settings.drain_within_days

        def live(keys: list[str]) -> list[Plan]:
            out = []
            for k in keys:
                p = self.plans.get(k)
                if p and p.enabled and not p.expired:
                    out.append(p)
            return out

        body = live(lane.order)
        # Expiring plans first, soonest death first, so the capacity with the
        # least time left gets drained first. Everything else keeps its
        # configured position (the sort is stable).
        body.sort(key=lambda p: (0, p.days_left)
                  if (p.days_left is not None and p.days_left <= window) else (1, 0))
        return body + live(lane.tail)

    def is_tail(self, lane_key: str, plan_key: str) -> bool:
        return plan_key in self.lanes[lane_key].tail


def _parse_probe(raw: Any) -> Probe | None:
    if not raw:
        return None
    body = dict(raw)
    fields = {k: (v if isinstance(v, list) else [v])
              for k, v in (body.pop("fields", None) or {}).items()}
    return Probe(fields=fields, **body)


def _parse_quotas(body: dict) -> tuple[Quota, ...]:
    """`quotas:` (a list) wins; `quota:` (a single mapping) still works.

    With several windows and none marked, the longest period becomes the target
    — maximising the weekly allowance is nearly always what you want, and the
    5-hour window is nearly always the thing not to overshoot.
    """
    raw = body.pop("quotas", None)
    if raw is None:
        single = dict(body.pop("quota", None) or {})
        return (Quota(**single),) if single else (Quota(),)

    order = {"rolling_5h": 0, "day": 1, "week": 2, "month": 3, None: 4}
    parsed = [Quota(**dict(entry)) for entry in raw]
    if not any(q.is_target for q in parsed):
        longest = max(parsed, key=lambda q: order.get(q.period, 4))
        parsed = [
            Quota(**{**q.__dict__, "role": "target" if q is longest else "constraint"})
            for q in parsed
        ]
    elif sum(1 for q in parsed if q.is_target) > 1:
        raise ValueError(f"only one quota window may be role: target, got {raw}")
    return tuple(parsed)


def _parse_date(v: Any) -> date | None:
    if v in (None, ""):
        return None
    if isinstance(v, date):
        return v
    return datetime.strptime(str(v), "%Y-%m-%d").date()


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
        probe_raw = body.pop("probe", None)
        raw_parallel = body.get("max_parallel", 1)
        auto = str(raw_parallel).lower() == "auto"
        configured = None if auto else int(raw_parallel)
        plans[key] = Plan(
            key=key,
            label=body.get("label", key),
            deployment=body["deployment"],
            max_parallel=configured if configured is not None else settings.concurrency_learning.seed_cap,
            configured_parallel=configured,
            max_parallel_ceiling=(int(body["max_parallel_ceiling"])
                                  if body.get("max_parallel_ceiling") else None),
            pacing=body.get("pacing"),
            supports_tools=body.get("supports_tools"),
            context_window=(int(body["context_window"])
                            if body.get("context_window") else None),
            monthly_cost=float(body.get("monthly_cost", 0) or 0),
            auth=body.get("auth", "api_key"),
            provider_family=body.get("provider_family"),
            subscription_key=body.get("subscription"),
            metered=bool(body.get("metered", False)),
            enabled=bool(body.get("enabled", True)),
            expires=_parse_date(body.get("expires")),
            quotas=quotas,
            probe=_parse_probe(probe_raw),
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

    return Registry(settings=settings, plans=plans, lanes=lanes)
