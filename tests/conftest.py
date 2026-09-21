"""Tests never talk to a real provider. This makes that enforceable.

The suite is meant to be runnable by anyone, on any machine, with no
credentials, no Docker and no subscriptions — and to spend nothing when it
runs. A test that quietly called api.x.ai or a local Ollama would break all of
that: it would fail for a contributor who has neither, and on the author's
machine it would burn real capacity to assert something a stub could prove.

So outbound sockets are blocked here, with loopback allowed: several tests do
start a stub HTTP server or a subprocess bridge on 127.0.0.1, which is the
point — they exercise the real protocol against a fake peer.

If you need to test against a provider, that belongs in scripts/smoke.py, which
checks a running deployment and says plainly that it spends quota.
"""
from __future__ import annotations

import socket

_LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0", ""}
_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


def _host_of(address) -> str:
    if isinstance(address, tuple) and address:
        return str(address[0])
    return str(address)


def _blocked(address) -> AssertionError:
    return AssertionError(
        f"a test tried to reach {_host_of(address)!r}. Tests must not call real "
        "providers — use a stub on 127.0.0.1, or put it in scripts/smoke.py.")


def _guarded_connect(self, address):
    if _host_of(address) not in _LOOPBACK:
        raise _blocked(address)
    return _real_connect(self, address)


def _guarded_connect_ex(self, address):
    if _host_of(address) not in _LOOPBACK:
        raise _blocked(address)
    return _real_connect_ex(self, address)


socket.socket.connect = _guarded_connect
socket.socket.connect_ex = _guarded_connect_ex
