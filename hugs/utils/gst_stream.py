"""
Live-streaming client for HUGS animation frames.

This module has NO GStreamer/PyGObject dependency on purpose: the `hugs`
conda env does not have PyGObject installed (and installing it needs system
dev headers we don't want to add as a side effect). Instead, this is a thin
stdlib-only TCP client that sends raw RGB frames to a separate server
process (`scripts/gst_stream_server.py`) which owns the actual GStreamer
pipeline and runs in the `gstreamer` conda env.

The status text burned into the top-left corner of the stream (LIVE
progress / STREAM COMPLETE / REPLAY) is drawn server-side via GStreamer's
`textoverlay` element (see WebrtcFeeder in gst_stream_server.py) — the
server is what actually knows whether it's forwarding a live frame or
looping a finished clip, so it owns the label. This client only needs to
tell it how many frames to expect, via `total_frames`.

Wire protocol (little care needed, this is a private local protocol):
  1. Client connects, sends one header line: JSON + b"\n"
     {"width": W, "height": H, "fps": FPS, "out_dir": ..., "segment_duration": S,
      "total_frames": N}
  2. For each frame: 4-byte big-endian uint32 length, followed by that many
     raw RGB24 bytes (row-major, no padding, HWC uint8).
  3. End of stream: a 4-byte length of 0 (no payload), then the socket is
     closed by the client after reading the server's ack byte.

Because the server does the (slow-ish) encode/mux/segment work in a
separate OS process, `push_frame()` here is just a `sendall()` — the
render loop and the encode/stream pipeline run concurrently for free.
"""

import json
import socket
import sys
from pathlib import Path

import torch

# scripts/pipeline_bus.py holds the shared (stdlib-only) low-level framing
# primitives used by both this client and scripts/gst_stream_server.py, so
# there's one wire format implementation, not two.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from pipeline_bus import _send_framed  # noqa: E402


class FrameStreamClient:
    def __init__(
        self,
        image: torch.Tensor,
        fps: int = 20,
        out_dir: str = "",
        segment_duration: float = 1.0,
        host: str = "127.0.0.1",
        port: int = 9977,
        connect_timeout: float = 10.0,
        total_frames: int = None,
    ):
        """`image` is a sample CHW float tensor in [0, 1] used only to read
        (height, width) so the server can set up its pipeline caps.

        total_frames: how many frames this session will push, if known
        upfront — used only for the server's "LIVE - rendering frame N/Total"
        status overlay (--mode webrtc). Omit if unknown."""
        _, height, width = image.shape
        self.width = width
        self.height = height

        self._sock = socket.create_connection((host, port), timeout=connect_timeout)
        self._sock.settimeout(None)

        header = {
            "width": width,
            "height": height,
            "fps": fps,
            "out_dir": out_dir,
            "segment_duration": segment_duration,
            "total_frames": total_frames,
        }
        self._sock.sendall((json.dumps(header) + "\n").encode("utf-8"))

    def push_frame(self, image: torch.Tensor) -> None:
        """image: CHW float tensor in [0, 1] (same shape as at construction time)."""
        frame = (image.clamp(0, 1) * 255).to(torch.uint8)
        frame = frame.permute(1, 2, 0).contiguous().cpu().numpy()  # HWC uint8
        assert frame.shape == (self.height, self.width, 3), (
            f"frame shape {frame.shape} != expected {(self.height, self.width, 3)}"
        )
        self._send(frame.tobytes())

    def _send(self, payload: bytes) -> None:
        _send_framed(self._sock, payload)

    def close(self) -> None:
        try:
            self._send(b"")  # zero-length frame == EOS marker
            self._sock.recv(1)  # wait for server's "done flushing" ack
        finally:
            self._sock.close()


def notify_pending(host: str = "127.0.0.1", port: int = 9977, timeout: float = 3.0) -> None:
    """Fire-and-forget ping telling the stream server a new run is about to
    start rendering — call this as early as possible (before MDM sampling,
    not just before the HUGS render loop), so viewers immediately see a
    "please wait, rendering..." placeholder instead of either the previous
    run's stale replay loop or nothing while MDM/SMPL-extraction run.

    Safe to call even if no server is listening (--stream-live off, or the
    server just isn't up): connection errors are swallowed silently, same
    as this being a best-effort UX nicety, not a required step."""
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        try:
            sock.sendall((json.dumps({"pending": True}) + "\n").encode("utf-8"))
        finally:
            sock.close()
    except OSError:
        pass
