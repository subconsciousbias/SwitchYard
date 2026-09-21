"""Load a sidecar's server.py under a unique module name.

All three sidecars name their entry point `server.py`. A plain `import server`
after a `sys.path.insert` therefore resolves to whichever one was imported
first: run alone each suite passed, run together two of the three tested a
module they had not loaded, with failures that read like real bugs. Loading by
path under an explicit name keeps them distinct, and keeps `importlib.reload`
working, since a spec-loaded module carries its own __spec__.
"""
from __future__ import annotations

import importlib.util
import sys


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
