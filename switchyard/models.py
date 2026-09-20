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


@dataclass(frozen=True)
class Lane:
    key: str
    label: str
    order: list[str]
    tail: list[str] = field(default_factory=list)
    description: str = ""


@dataclass(frozen=True)
class Settings:
    drain_within_days: int = 21
    lease_ttl_seconds: int = 1800
    inflight_max_age_seconds: int = 900
    default_cooldown_seconds: int = 900


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
        # Stable sort: expiring-soon first, otherwise keep configured position.
        body.sort(key=lambda p: 0 if (p.days_left is not None and p.days_left <= window) else 1)
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

    settings = Settings(**(raw.get("settings") or {}))

    plans: dict[str, Plan] = {}
    for key, body in (raw.get("plans") or {}).items():
        body = dict(body)
        q = dict(body.pop("quota", None) or {})
        plans[key] = Plan(
            key=key,
            label=body.get("label", key),
            deployment=body["deployment"],
            max_parallel=int(body.get("max_parallel", 1)),
            monthly_cost=float(body.get("monthly_cost", 0) or 0),
            auth=body.get("auth", "api_key"),
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
