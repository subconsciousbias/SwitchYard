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
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Union

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
    # Move inline <think>...</think> out of streamed content and into
    # reasoning_content, the way a buffered response already does.
    split_reasoning_tags: bool = True
    inflight_max_age_seconds: int = 120
    heartbeat_seconds: int = 30
    default_cooldown_seconds: int = 900
    concurrency_learning: ConcurrencyLearning = field(default_factory=ConcurrencyLearning)
    pacing: Pacing = field(default_factory=Pacing)
    transient_breaker: TransientBreaker = field(default_factory=TransientBreaker)


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

        Two rules on top of the configured order: dead members drop out (plan
        disabled or expired, or the model disabled), and a member whose plan
        expires within `drain_within_days` is promoted ahead of members on plans
        that are not expiring, soonest death first, so cancelled capacity gets
        used before it vanishes. Tail members always stay last.

        Group structure is collapsed before filtering: a `{round_robin: [a,
        b]}` and the flat `[a, b]` produce the same effective ordering, and a
        `{weighted: {a: 5, b: 2}}` walks a then b in declaration order — the
        weighting is the picker's job, not the body's.
        """
        refs = self.routing_order(lane_key)
        window = self.settings.drain_within_days

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

        body = live(refs)

        def urgency(model: Model) -> tuple[int, int]:
            days = self.plans[model.plan_key].days_left
            return (0, days) if (days is not None and days <= window) else (1, 0)

        body.sort(key=urgency)
        return body + live(self.lanes[lane_key].tail)

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
        if depth >= _MAX_GROUP_DEPTH:
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


def load(path: str | None = None) -> Registry:
    with open(path or CONFIG_PATH) as fh:
        raw = yaml.safe_load(fh)

    sraw = dict(raw.get("settings") or {})
    settings = Settings(
        **{k: v for k, v in sraw.items() if k not in (
            "concurrency_learning", "pacing", "transient_breaker")},
        concurrency_learning=ConcurrencyLearning(**(sraw.get("concurrency_learning") or {})),
        pacing=Pacing(**(sraw.get("pacing") or {})),
        transient_breaker=TransientBreaker(**(sraw.get("transient_breaker") or {})),
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
        order = _parse_lane_order(key, list(body.get("order") or []), known)
        lanes[key] = Lane(
            key=key,
            label=body.get("label", key),
            order=order,
            tail=list(body.get("tail") or []),
            description=body.get("description", ""),
            strategy=strategy,
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
    return registry
