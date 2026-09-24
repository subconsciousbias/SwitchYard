"""Load a sidecar's server.py under a unique module name.

All three sidecars name their entry point `server.py`. A plain `import server`
after a `sys.path.insert` therefore resolves to whichever one was imported
first: run alone each suite passed, run together two of the three tested a
module they had not loaded, with failures that read like real bugs. Loading by
path under an explicit name keeps them distinct, and keeps `importlib.reload`
working, since a spec-loaded module carries its own __spec__.
"""
from __future__ import annotations

import conftest  # noqa: F401  (socket guard)

import importlib.util
import os
import sys

# The bridges generate codex's locked-down model catalog by running
# `codex debug models` (cli_bridge.codex_catalog_path). There is no codex
# binary offline, so point every bridge loaded in a test at a fake that
# answers exactly that command. Set before any bridge is loaded: the codex
# profile reads CODEX_CLI at import.
os.environ["CODEX_CLI"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures", "fake_codex.py")


def load(name: str, path: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod              # set first: the module may import itself
    spec.loader.exec_module(mod)
    return mod


def reload(mod):
    """Re-execute a module loaded by `load`.

    importlib.reload() re-*finds* the spec by module name, which fails for a
    name that exists nowhere on sys.path. The spec we built is still attached,
    so run it again against the same module object.
    """
    mod.__spec__.loader.exec_module(mod)
    return mod


class FakePoppler:
    """pdftotext / pdftoppm stand-ins on PATH: deterministic text, two pages.

    `text_fails` / `pages_fail` make that half exit 1 with nothing produced
    (both -> a PDF poppler can make nothing of)."""

    def __init__(self, text_fails: bool = False, pages_fail: bool = False):
        self.text_fails = text_fails
        self.pages_fail = pages_fail

    def __enter__(self):
        import stat
        import tempfile
        from pathlib import Path
        self.dir = Path(tempfile.mkdtemp(prefix="fake-poppler-"))
        (self.dir / "pdftotext").write_text(
            "#!/bin/sh\necho broken >&2; exit 1\n" if self.text_fails
            else "#!/bin/sh\necho EXTRACTED TEXT\n")
        (self.dir / "pdftoppm").write_text(
            "#!/bin/sh\necho unrenderable >&2; exit 1\n" if self.pages_fail
            else "#!/bin/sh\nfor last; do :; done\n"
                 "printf PNG1 > \"$last-1.png\"; printf PNG2 > \"$last-2.png\"\n")
        for tool in ("pdftotext", "pdftoppm"):
            (self.dir / tool).chmod(stat.S_IRWXU)
        self.path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{self.dir}:{self.path}"
        return self

    def __exit__(self, *exc):
        import shutil
        os.environ["PATH"] = self.path
        shutil.rmtree(self.dir, ignore_errors=True)
