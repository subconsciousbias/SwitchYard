# Working in `switchyard/portal/`

The portal: capacity board, quota headroom, subscription economics. Runs in
the `portal` image, baked from `Dockerfile.portal`.

## Board/gateway parity rule

Cap / headroom / ranking math has **one home** in `switchyard/`:

  - `switchyard.usage.headroom` (the per-plan quota headroom the board shows).
  - `switchyard.models.cap_for` (the per-model cap the picker / litellm
    config / board all read).
  - `switchyard.picker.Picker` (the lane ranking the picker applies to choose
    a model for a request).

When any of these change — a new quota field, a new scoring rule, a new cap
shape — re-check the board:

  - `switchyard/portal/app.py` imports `headroom` from `switchyard.usage`
    (`app.py:29`) and renders it for every plan row (`app.py:545`, `581`).
  - `switchyard/portal/app.py` calls `plan.cap_for(m, reg.settings)` to render
    the per-row cap (`app.py:632`).
  - The board reads the same Redis hashes (`sy:lane-order:{lane}`,
    `sy:group-order:{gid}:{lane}`) the picker reads, via `get_lane_order` /
    `get_group_order`, so the displayed order is whatever the picker would
    walk.

This is not enforced mechanically; prose is the interim rule. The shape
that drift takes in practice is a *portal-local* copy of a piece of math
that should have come from `switchyard/`. Don't re-derive those. If a
board-specific presentation is needed (rounding, units, captions), do it
in the template or in a portal-local render helper that calls into the
canonical function — not by copying the math.

`switchyard/portal/groups.py` is the per-group writer. Its
`_partition_with_tier`, `_TIER_OFFSET`, and `_one_adjacent_swap_partitioned`
mirror the lane-level writer in `app.py` by import — see the lock-step
note at `switchyard/portal/groups.py:129` — so a regression in
either writer is a regression in the other. Edit them together.

## Template conventions

Jinja2 templates live in `switchyard/portal/templates/`:

  - `base.html` is the shared layout — every other template extends it.
    Header, footer, navigation, the global alert banner. Add to `base.html`
    when a new page-level element is needed everywhere; add to the
    page-specific fragment otherwise.
  - `index.html` is the page the portal serves at `/`. It includes
    `_pacing.html`, `_capacity.html`, `_probes.html`, and `_plans.html`
    for the four panels.
  - `_capacity.html`, `_capacity_slots.html`, `_capacity_state.html` are the
    capacity-board fragments.
  - `_plans.html` is the plans / headroom panel.
  - `_pacing.html` is the pacing strip.
  - `_probes.html` is the cookie-needing-probes panel.

`tests/render_preview.py` renders every fragment against a synthetic
two-window fixture so Jinja errors and layout regressions get caught
without Redis, the gateway or any provider. New templates belong in
`tests/render_preview.py`'s fragment list so they get the same exercise.
