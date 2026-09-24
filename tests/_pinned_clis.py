"""The vendor CLIs pinned in Dockerfile.sidecar, for the lockdown tests.

tests/test_lockdown_*.py run each sidecar CLI -- the exact version the image
pins -- against tests/_fake_model.py, to prove the bridges' argv really keeps
the CLI's own tools, prompt sections and login-dir files away from the model
(issue #264). A CLI release is where that has broken every time, so the pins
are read from the Dockerfile itself: one source of truth for what ships.

Locally the CLIs are usually absent and those tests skip. CI's `cli-lockdown`
job installs the pinned versions and sets SWITCHYARD_REQUIRE_CLIS=1, which
turns a missing or mismatched CLI into a failure instead of a skip.

`python3 tests/_pinned_clis.py` prints the `npm install -g` package list.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

DOCKERFILE = Path(__file__).resolve().parent.parent / "Dockerfile.sidecar"
PACKAGES = {"claude": "@anthropic-ai/claude-code", "codex": "@openai/codex",
            "opencode": "opencode-ai"}
REQUIRED = os.environ.get("SWITCHYARD_REQUIRE_CLIS") == "1"


class Skip(Exception):
    """Raised (locally) when a pinned CLI is not installed."""


def pinned_versions() -> dict[str, str]:
    """{"claude": "2.1.278", ...} from the Dockerfile's npm install line."""
    text = DOCKERFILE.read_text()
    found = {}
    for cli, package in PACKAGES.items():
        m = re.search(rf"{re.escape(package)}@([0-9][^\s\\]*)", text)
        if not m:
            raise AssertionError(f"{package} is not pinned in {DOCKERFILE.name}")
        found[cli] = m.group(1)
    return found


def installed_version(cli: str) -> str | None:
    path = shutil.which(cli)
    if not path:
        return None
    out = subprocess.run([path, "--version"], capture_output=True, text=True,
                         timeout=60).stdout
    m = re.search(r"\d+\.\d+\.\d+", out)
    return m.group(0) if m else out.strip()


def require(cli: str) -> str:
    """Path to the pinned `cli`. Skip locally when absent; fail in CI when
    absent or when its version is not the one Dockerfile.sidecar pins."""
    want = pinned_versions()[cli]
    have = installed_version(cli)
    if have is None:
        if REQUIRED:
            raise AssertionError(f"{cli} {want} is required but not installed")
        raise Skip(f"{cli} not installed (CI installs {PACKAGES[cli]}@{want})")
    if have != want:
        message = f"{cli} {have} is installed; Dockerfile.sidecar pins {want}"
        if REQUIRED:
            raise AssertionError(message)
        raise Skip(message)
    return shutil.which(cli)


if __name__ == "__main__":
    print(" ".join(f"{PACKAGES[c]}@{v}" for c, v in pinned_versions().items()))
