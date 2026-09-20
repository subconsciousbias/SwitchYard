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
    kind: str = "unknown"          # tokens | dollars | window | unlimited | unknown
    period: str | None = None      # month | week | rolling_5h
    allowance: float | None = None
    source: str = "estimate"       # estimate | headers | probe | ledger | sidecar | none
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Plan:
    key: str
    label: str
    deployment: str
    max_parallel: int
    monthly_cost: float = 0.0
    auth: str = "api_key"
    provider_family: str | None = None   # drives vendor error-code mapping
    configured_parallel: int | None = None   # None when `max_parallel: auto`
    max_parallel_ceiling: int | None = None  # hard upper bound for learning
    pacing: bool | None = None               # per-plan override of the global switch
    metered: bool = False
    enabled: bool = True
    expires: date | None = None
    quota: Quota = field(default_factory=Quota)
    alert_burn_rate_per_hour: float | None = None
    notes: str = ""

    @property
    def expired(self) -> bool:
        return self.expires is not None and self.expires < date.today()

    @property
    def days_left(self) -> int | None:
        if self.expires is None:
            return None
        return (self.expires - date.today()).days

    @property
    def is_subscription(self) -> bool:
        """A fixed-fee plan with a resetting allowance — the only kind worth
        pacing. Metered providers and unlimited local models are not."""
        return not self.metered and self.quota.kind not in ("unlimited", "unknown")

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
        q = dict(body.pop("quota", None) or {})
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
            monthly_cost=float(body.get("monthly_cost", 0) or 0),
            auth=body.get("auth", "api_key"),
            provider_family=body.get("provider_family"),
            metered=bool(body.get("metered", False)),
            enabled=bool(body.get("enabled", True)),
            expires=_parse_date(body.get("expires")),
            quota=Quota(**q),
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
