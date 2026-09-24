#!/usr/bin/env python3
"""Stand-in for the `codex` binary, answering only `codex debug models`.

The bridges build codex's argv with a locked-down model catalog generated
from `codex debug models` (cli_bridge.codex_catalog_path). The offline suite
has no codex binary, so tests point CODEX_CLI here (see tests/_modules.py).
The catalog shape mirrors the pinned 0.155.1: a code-mode entry with every
field the lockdown strips, and a plain one.
"""
import json
import sys

if sys.argv[1:3] != ["debug", "models"]:
    sys.stderr.write(f"fake codex: only `debug models` is supported, got {sys.argv[1:]}\n")
    raise SystemExit(2)

print(json.dumps({"models": [
    {"slug": "gpt-5.6-terra", "shell_type": "unified_exec",
     "tool_mode": "code_mode_only", "multi_agent_version": "v2",
     "apply_patch_tool_type": "freeform", "web_search_tool_type": "text_and_image",
     "supports_search_tool": True, "experimental_supported_tools": ["clock"]},
    {"slug": "gpt-5.5", "shell_type": "unified_exec",
     "apply_patch_tool_type": "freeform", "supports_search_tool": True,
     "experimental_supported_tools": []},
]}))
