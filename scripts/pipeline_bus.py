#!/usr/bin/env python3
"""
Stdlib-only client for the pipeline bus: generic push/pull data hand-off
between pipeline stages that run in different conda envs (mdm, hugs), routed
through scripts/gst_stream_server.py -- the "broker", the only process with
PyGObject/GStreamer installed (see that file for the server-side PipelineBus
implementation). No numpy/torch/gi import here on purpose: this module must
import cleanly from every env a pipeline stage might run in.

Shares its wire format with hugs/utils/gst_stream.py's video-frame protocol:
one JSON header line (`\\n`-terminated) followed by one or more length-prefixed
frames (4-byte big-endian uint32 length, then that many raw bytes; a
zero-length frame marks end-of-stream). This module is where those low-level
framing primitives (_send_framed / _recv_exact) live -- gst_stream.py and
gst_stream_server.py both import them from here instead of each keeping their
own copy.

Bus operations (one JSON header per connection, "bus_op" selects the verb):
  {"bus_op": "ping"}                                      -> one ack byte back
  {"bus_op": "push", "run_id", "stage"} + framed payload   -> one ack byte back
  {"bus_op": "pull", "run_id", "stage", "timeout"}         -> framed payload back,
      or a BUS_ERROR_MARKER length prefix + framed JSON {"error": ...} on timeout

By default, each pipeline stage hands data to the next purely through the bus
-- no intermediate .npy/.npz files on disk. Pass archive=True/archive_path=...
to push() to *also* keep the conventional on-disk copy (this is what
--save-intermediate wires up in scripts/run_text2hugs.py).
"""

import json
import socket
import struct
from pathlib import Path
from typing import Optional

BUS_ERROR_MARKER = 0xFFFFFFFF  # reserved length value: "next frame is a JSON error, not payload"


class BusError(RuntimeError):
    """Raised for any bus-level failure (connection, protocol, server-side error)."""


class BusTimeoutError(BusError):
    """Raised when pull() times out waiting for a producer's push()."""


class _BusErrorMarker(Exception):
    """Internal sentinel: the frame just read was BUS_ERROR_MARKER, meaning the
    next frame on the wire is a JSON error message, not real payload data."""


def _send_framed(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(struct.pack(">I", len(payload)))
    if payload:
        sock.sendall(payload)


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def _recv_framed(sock: socket.socket) -> Optional[bytes]:
    """Reads one length-prefixed frame. Returns None on EOF/disconnect before
    any length prefix arrives, raises _BusErrorMarker if the length prefix is
    the BUS_ERROR_MARKER sentinel (caller then reads the following framed JSON
    error message itself)."""
    length_bytes = _recv_exact(sock, 4)
    if length_bytes is None:
        return None
    (length,) = struct.unpack(">I", length_bytes)
    if length == BUS_ERROR_MARKER:
        raise _BusErrorMarker()
    if length == 0:
        return b""
    return _recv_exact(sock, length)


def _send_header(sock: socket.socket, header: dict) -> None:
    sock.sendall((json.dumps(header) + "\n").encode("utf-8"))


def is_broker_alive(host: str = "127.0.0.1", port: int = 9977, timeout: float = 2.0) -> bool:
    """Best-effort liveness check for the pipeline bus / gst_stream_server.py broker."""
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            _send_header(sock, {"bus_op": "ping"})
            sock.settimeout(timeout)
            ack = sock.recv(1)
            return ack == b"\x01"
    except OSError:
        return False


def push(
    run_id: str,
    stage: str,
    payload: bytes,
    *,
    archive: bool = False,
    archive_path: Optional[Path] = None,
    host: str = "127.0.0.1",
    port: int = 9977,
    connect_timeout: float = 10.0,
) -> None:
    """Push `payload` bytes onto the bus keyed by (run_id, stage), for the next
    stage's pull() to consume.

    If archive=True and archive_path is given, ALSO writes payload to that
    path on the local filesystem -- client-side, not server-side, since the
    broker doesn't know this repo's directory conventions and each stage
    already has its own conventional --output path in scope. Used for
    --save-intermediate.
    """
    if archive and archive_path is not None:
        archive_path = Path(archive_path)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_bytes(payload)

    with socket.create_connection((host, port), timeout=connect_timeout) as sock:
        sock.settimeout(None)
        _send_header(sock, {"bus_op": "push", "run_id": run_id, "stage": stage})
        _send_framed(sock, payload)
        ack = sock.recv(1)
        if ack != b"\x01":
            raise BusError(f"push(run_id={run_id!r}, stage={stage!r}): broker did not ack")


def pull(
    run_id: str,
    stage: str,
    *,
    timeout: float = 60.0,
    host: str = "127.0.0.1",
    port: int = 9977,
    connect_timeout: float = 10.0,
) -> bytes:
    """Blocks until (run_id, stage)'s producer has push()ed, or `timeout`
    seconds elapse -- whichever comes first. Raises BusTimeoutError on
    timeout, BusError on any other bus/connection failure."""
    with socket.create_connection((host, port), timeout=timeout + connect_timeout) as sock:
        _send_header(sock, {"bus_op": "pull", "run_id": run_id, "stage": stage, "timeout": timeout})
        try:
            payload = _recv_framed(sock)
        except _BusErrorMarker:
            err_payload = _recv_framed(sock)
            message = {}
            if err_payload:
                try:
                    message = json.loads(err_payload.decode("utf-8"))
                except json.JSONDecodeError:
                    pass
            error_text = message.get("error", "unknown broker error")
            if "timeout" in error_text:
                raise BusTimeoutError(f"pull(run_id={run_id!r}, stage={stage!r}): {error_text}")
            raise BusError(f"pull(run_id={run_id!r}, stage={stage!r}): {error_text}")
        if payload is None:
            raise BusError(f"pull(run_id={run_id!r}, stage={stage!r}): connection closed before payload arrived")
        return payload


def push_file(
    run_id: str, stage: str, path, *, archive: bool = False, archive_path=None, **kw
) -> None:
    push(run_id, stage, Path(path).read_bytes(), archive=archive, archive_path=archive_path, **kw)


def pull_to_file(run_id: str, stage: str, path, **kw) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pull(run_id, stage, **kw))
    return path
