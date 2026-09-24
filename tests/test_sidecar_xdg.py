"""Regression tests for the Dockerfile.sidecar XDG-state-dir fix.

Issue #247 (opencode-go2 502 on every call): the sidecar image stopped
baking the node-owned XDG tree into the image. compose runs every
`*-sidecar` as `user: node` (uid 1000) so when the host bind-mounts land
on `/home/node/.local/share/opencode` and `/home/node/.config/opencode`
but the parent `/home/node/.local/{share,state}` and `/home/node/.cache`
do not exist, Docker materialises those missing parents as `root:root`.
`opencode --version` (and every subsequent call) then tries to mkdir its
`$XDG_STATE_HOME` (defaulting to `$HOME/.local/state`) and gets EACCES,
so the daemon 502s every request — without logging the gate's surface
because the failure happens inside the vendor CLI's startup.

Dockerfile.sidecar's fix is a single RUN that creates the four XDG dirs
(`/home/node/.local/state`, `/home/node/.local/share`, `/home/node/.config`,
`/home/node/.cache`), chowns the three parents to `node:node`, and pins
`XDG_STATE_HOME=/home/node/.local/state` in the same ENV block as the
other XDG vars. Deleting any of those four lines returns the sidecar to
the broken state — silently, because smoke.py also did not check
`opencode-go2-sidecar` (the blind spot the task spec calls out).

These three tests assert the fix is on disk as raw fragments of the
Dockerfile, the same raw-fragment style as
`tests/test_litellm_patch.py::test_dockerfile_contains_the_patch_block`:
substring-match the bytes the fix would have written, so a regression
fails the offline suite loudly instead of re-darkening a plan in
production. The tests are pure file/string work — no switchyard import,
no Docker, no network.
"""
from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SIDECAR = os.path.join(ROOT, "Dockerfile.sidecar")


def _read_sidecar() -> str:
    with open(SIDECAR, encoding="utf-8") as fh:
        return fh.read()


# The exact four dirs Dockerfile.sidecar must `mkdir -p` and chown onto.
# Listed here as a constant set so each test can assert on its own slice
# — the all-four assertion fails if any one line is dropped, but we also
# break it up so a partial regression names the missing dir at review
# time. The order matches the Dockerfile so a diff against the source
# file is also a diff against the test.
_XDG_DIRS = (
    "/home/node/.local/state",
    "/home/node/.local/share",
    "/home/node/.config",
    "/home/node/.cache",
)

# The three XDG parents whose ownership Dockerfile.sidecar must transfer
# from the root-created bind-mount materialisation to the running user.
# `.local/state` and `.local/share` share the `.local` parent; `.config`
# and `.cache` are their own roots.
_XDG_PARENTS = (
    "/home/node/.local",
    "/home/node/.config",
    "/home/node/.cache",
)


def test_dockerfile_sidecar_creates_all_four_xdg_dirs():
    """Dockerfile.sidecar's fix RUN must `mkdir -p` every XDG parent the
    vendor CLIs resolve against `$XDG_*_HOME`. Drop any one of them and
    the sidecar returns to the `#247` EACCES failure mode (compose
    materialises the missing dir as root and `user: node` cannot write
    to it). The assertion is a raw-fragment match, not an AST, so the
    test fails on the exact bytes the fix would have written — same
    style as test_litellm_patch.py's Dockerfile fragment check.
    """
    src = _read_sidecar()
    missing = [d for d in _XDG_DIRS if d not in src]
    assert not missing, (
        f"Dockerfile.sidecar is missing the following XDG dir(s) from "
        f"its fix RUN: {missing}. Without `mkdir -p` on each, the "
        f"`user: node` sidecar cannot materialise its XDG parents "
        f"under bind-mounts and `opencode --version` (and every "
        f"subsequent CLI call) hits EACCES — the issue #247 failure "
        f"mode."
    )
    # All four must appear together inside one `mkdir -p` invocation:
    # a fix that splits them across multiple RUNs is fragile (a build
    # that fails between two RUNs leaves an intermediate image) and
    # needlessly churns layers.
    assert "mkdir -p /home/node/.local/state /home/node/.local/share "\
           "/home/node/.config /home/node/.cache" in src, (
        "Dockerfile.sidecar does not `mkdir -p` the four XDG dirs "
        "together in a single invocation; a multi-RUN split would "
        "leave the broken state if a later RUN failed."
    )
    print(f"  Dockerfile.sidecar: mkdir -p of all {len(_XDG_DIRS)} "
          "XDG dirs present")


def test_dockerfile_sidecar_chowns_three_xdg_parents_to_node():
    """The same fix RUN must `chown -R node:node` on the three parents,
    so the dirs are writable by `user: node`. Without this, mkdir
    succeeds but the chown step is missing and the bind-mount still
    lands on a root-owned path — same EACCES failure mode, different
    reason. Substring-match on the exact `chown -R` shape the fix
    would have written.
    """
    src = _read_sidecar()
    missing = [p for p in _XDG_PARENTS if p not in src]
    assert not missing, (
        f"Dockerfile.sidecar is missing the following XDG parent(s) "
        f"from its chown step: {missing}. The bind-mount materialisation "
        f"is root-owned until something chowns it; without the chown, "
        f"`user: node` cannot write to the XDG parents and `opencode "
        f"--version` EACCESes again."
    )
    assert "chown -R node:node /home/node/.local /home/node/.config "\
           "/home/node/.cache" in src, (
        "Dockerfile.sidecar does not `chown -R node:node` all three "
        "XDG parents together in one invocation; a partial chown "
        "leaves one or more dirs root-owned and re-darkens the plan."
    )
    print(f"  Dockerfile.sidecar: chown -R node:node of all "
          f"{len(_XDG_PARENTS)} XDG parents present")


def test_dockerfile_sidecar_pins_xdg_state_home_env():
    """The `XDG_STATE_HOME` env must be pinned to `/home/node/.local/state`
    in the sidecar's ENV block, alongside the other XDG vars. Without
    it the vendor CLI's default (`$HOME/.local/state`) lands on the
    root-owned bind-mount path and the same EACCES fires — distinct
    from the mkdir/chown fix, but the same observable outage.
    """
    src = _read_sidecar()
    assert "XDG_STATE_HOME=/home/node/.local/state" in src, (
        "Dockerfile.sidecar is missing "
        "`XDG_STATE_HOME=/home/node/.local/state` in its ENV block. "
        "Without the explicit pin the vendor CLI's default resolves to "
        "`$HOME/.local/state`, which on a recreated image lands on the "
        "root-owned bind-mount parent and EACCESes again — the same "
        "502 every-call failure the mkdir/chown fix is meant to close."
    )
    print("  Dockerfile.sidecar: XDG_STATE_HOME=/home/node/.local/state "
          "is pinned in the ENV block")


if __name__ == "__main__":
    import _runner
    raise SystemExit(_runner.run(globals()))