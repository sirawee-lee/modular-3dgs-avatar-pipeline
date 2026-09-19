#!/usr/bin/env python3
"""
GStreamer live-streaming server for HUGS animation frames.

Run this with the `gstreamer` conda env's Python (it needs PyGObject and
GStreamer's nvcodec/hls plugins, which the `hugs` env does not have):

  /home/sigma/anaconda3/envs/gstreamer/bin/python scripts/gst_stream_server.py \
      --port 9977

Accepts ONE connection at a time from hugs/utils/gst_stream.py, reads a
JSON header line describing the frame size/fps, then raw RGB24 frames
(4-byte big-endian length prefix + payload; a zero-length payload marks
end-of-stream). Frames are pushed into an `appsrc` and encoded with NVENC
H.264 (falls back to software vp8 if no NVIDIA GPU is available). What
happens after encoding depends on --mode:

  --mode hls (default): segmented into HLS (playlist.m3u8 + .ts chunks)
    that a browser/VLC can start playing before the whole run finishes
    rendering. One pipeline per connection, torn down when the session ends.
      appsrc ! queue ! videoconvert ! nvh264enc ! h264parse ! hlssink2

  --mode webrtc: pushed via RTSP RECORD to a MediaMTX instance (see
    tools/mediamtx/mediamtx.yml), which re-exposes it as WebRTC and does
    all the SDP/ICE/DTLS-SRTP work — no webrtcbin here. This env's
    gst-plugins-bad build has no libnice, so webrtcbin itself isn't
    available; MediaMTX is what makes real (sub-second) WebRTC delivery
    possible without it.
      appsrc ! queue ! videoconvert ! nvh264enc ! h264parse ! rtspclientsink

    The pipeline is built ONCE and kept running for the server's whole
    lifetime (not torn down between connections): a HUGS render session
    is typically only ~5-15s of frames, and MediaMTX/WebRTC only shows
    *something* to viewers while a publisher is actively connected — if we
    tore the RTSP publish down at each session's end, the stream would go
    dark ("stream not found") the moment rendering finished, and anyone
    who opened the viewer page a few seconds late would miss it entirely.
    Instead, once a render session ends, the last completed clip is looped
    (replayed frame-by-frame at the original fps) until the next session
    connects and takes over live. See WebrtcFeeder below.

    A `textoverlay` element burns a live status string into the top-left
    corner: "LIVE - rendering frame N/Total" while a render session is
    actively pushing, "STREAM COMPLETE - N frames" for a couple of seconds
    right after it ends, then "REPLAY (idle) - frame N/Total" while looping.
    The label is computed server-side (WebrtcFeeder._push) since only the
    server actually knows whether a given frame is live or being replayed —
    the render process just pushes raw pixels, no overlay/text logic there.

  In both modes, `queue` decouples the socket-reading loop from encode/mux
  so the render process's push_frame() calls never block on encoding.
"""

import argparse
import json
import os
import socket
import struct
import threading
import time

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import GLib, Gst, GstVideo  # noqa: E402


def _wrap_frame(payload: bytes, width: int, height: int) -> Gst.Buffer:
    """Wrap a tightly-packed RGB24 frame (row stride == width*3, as sent by
    hugs/utils/gst_stream.py) into a Gst.Buffer with explicit video meta.

    Without this, downstream elements assume GStreamer's default raw-video
    row stride, which is width*3 rounded UP to a multiple of 4 — for any
    width where width*3 isn't already a multiple of 4 (e.g. 1266 -> 3798,
    padded to 3800), that mismatch silently desyncs every row and the
    decoded image degrades to black well before the bottom of the frame.
    """
    buf = Gst.Buffer.new_wrapped(payload)
    # PyGObject's binding wants fixed-size (GST_VIDEO_MAX_PLANES == 4) arrays
    # regardless of n_planes; unused trailing entries are ignored.
    GstVideo.buffer_add_video_meta_full(
        buf, GstVideo.VideoFrameFlags.NONE, GstVideo.VideoFormat.RGB,
        width, height, 1, [0, 0, 0, 0], [width * 3, 0, 0, 0],
    )
    return buf


# ---------------------------------------------------------------------------
# --mode hls: one pipeline per connection (unchanged from the original design)
# ---------------------------------------------------------------------------

def build_hls_pipeline(width: int, height: int, fps: int, out_dir: str, segment_duration: float) -> Gst.Pipeline:
    os.makedirs(out_dir, exist_ok=True)

    have_nvenc = Gst.ElementFactory.find("nvh264enc") is not None
    if have_nvenc:
        encoder = "nvh264enc ! h264parse config-interval=-1"
        target_duration = max(1, round(segment_duration))
        sink = (
            f"hlssink2 name=sink target-duration={target_duration} max-files=0 "
            f'playlist-location="{out_dir}/playlist.m3u8" location="{out_dir}/segment%05d.ts"'
        )
    else:
        # Software fallback: no x264/openh264/nvenc on this box -> vp8 + local
        # chunked .webm files via splitmuxsink (HLS needs an h264 encoder).
        encoder = "vp8enc deadline=1 cpu-used=8"
        max_size_time = int(segment_duration * Gst.SECOND)
        sink = (
            f'splitmuxsink name=sink muxer=webmmux max-size-time={max_size_time} '
            f'location="{out_dir}/segment%05d.webm"'
        )

    pipeline_str = (
        f"appsrc name=src is-live=true block=true format=time "
        f"caps=video/x-raw,format=RGB,width={width},height={height},framerate={fps}/1 "
        f"! queue name=q max-size-buffers=8 leaky=no "
        f"! videoconvert ! {encoder} ! {sink}"
    )
    print(f"[gst_stream_server] pipeline: {pipeline_str}")
    return Gst.parse_launch(pipeline_str)


def handle_hls_connection(conn: socket.socket, header: dict) -> None:
    width, height, fps = header["width"], header["height"], header["fps"]
    out_dir = header["out_dir"]
    segment_duration = header.get("segment_duration", 1.0)
    frame_size = width * height * 3

    print(f"[gst_stream_server] session start: {width}x{height}@{fps}fps -> {out_dir} (hls)")

    pipeline = build_hls_pipeline(width, height, fps, out_dir, segment_duration)
    appsrc = pipeline.get_by_name("src")

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_message(_bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"[gst_stream_server] pipeline error: {err} ({debug})")
            loop.quit()

    bus.connect("message", on_message)
    glib_thread = threading.Thread(target=loop.run, daemon=True)

    pipeline.set_state(Gst.State.PLAYING)
    glib_thread.start()

    frame_idx = 0
    fps_num = Gst.SECOND // fps
    try:
        while True:
            length_bytes = _recv_exact(conn, 4)
            if length_bytes is None:
                print("[gst_stream_server] client disconnected mid-stream")
                break
            (length,) = struct.unpack(">I", length_bytes)
            if length == 0:
                break  # EOS marker
            if length != frame_size:
                raise ValueError(f"expected {frame_size} bytes, got {length}")
            payload = _recv_exact(conn, length)
            if payload is None:
                break

            buf = _wrap_frame(payload, width, height)
            buf.pts = frame_idx * fps_num
            buf.duration = fps_num
            appsrc.emit("push-buffer", buf)
            frame_idx += 1
    finally:
        appsrc.emit("end-of-stream")
        glib_thread.join(timeout=30)
        pipeline.set_state(Gst.State.NULL)
        conn.sendall(b"\x01")  # ack: fully flushed
        print(f"[gst_stream_server] session done: {frame_idx} frames -> {out_dir}")


# ---------------------------------------------------------------------------
# --mode webrtc: one persistent pipeline for the server's whole lifetime,
# looping the last completed clip between render sessions.
# ---------------------------------------------------------------------------

def build_webrtc_pipeline(width: int, height: int, fps: int, mediamtx_url: str) -> Gst.Pipeline:
    have_nvenc = Gst.ElementFactory.find("nvh264enc") is not None
    # MediaMTX handles WebRTC delivery to the browser; we just need to get an
    # H.264 (or VP8) elementary stream to it via RTSP RECORD. rtspclientsink
    # auto-selects a payloader from the input caps.
    if have_nvenc:
        encoder = "nvh264enc ! h264parse config-interval=-1"
    else:
        encoder = "vp8enc deadline=1 cpu-used=8"
    sink = f'rtspclientsink name=sink location="{mediamtx_url}" protocols=udp+tcp'

    # textoverlay burns a status string (set at runtime via WebrtcFeeder._push,
    # e.g. "LIVE - rendering frame N/Total") into the top-left corner, so any
    # viewer sees it — no browser-side code or side channel needed.
    # videobalance's brightness is spiked for a couple of ticks whenever a
    # genuinely new live frame arrives (WebrtcFeeder._push), producing a
    # brief full-frame flash so a viewer can see exactly when new data
    # landed, as opposed to a held/repeated tick of the same frame.
    pipeline_str = (
        f"appsrc name=src is-live=true block=true format=time "
        f"caps=video/x-raw,format=RGB,width={width},height={height},framerate={fps}/1 "
        f"! queue name=q max-size-buffers=8 leaky=no "
        f"! videoconvert "
        f"! videobalance name=balance brightness=0 "
        f"! textoverlay name=overlay text=\"\" halignment=left valignment=top "
        f"  font-desc=\"Sans Bold 20\" color=0xFF39FF14 shaded-background=true "
        f"! videoconvert ! {encoder} ! {sink}"
    )
    print(f"[gst_stream_server] pipeline: {pipeline_str}")
    return Gst.parse_launch(pipeline_str)


class WebrtcFeeder:
    """Keeps a single RTSP-to-MediaMTX publish alive across render sessions,
    fed by a steady ticker rather than pushed straight from the network.

    HUGS rendering only produces frames at ~2/s (real GPU render time per
    frame), far below the stream's declared 20fps. Pushing each frame to
    the encoder the instant it arrives would timestamp it as if frames were
    50ms apart when they're really ~500ms apart — the encoded stream would
    either claim to run 10x faster than the real motion, or (since nothing
    else paces it) just show long freezes followed by a jump to whatever
    arrived meanwhile. Neither is "smooth".

    Instead, a background thread ticks at a fixed 1/fps interval and always
    pushes whatever the most-recently-known frame is: a newly rendered live
    frame if one has arrived since the last tick, otherwise the same frame
    again (held). This is the standard way to feed a fixed-framerate live
    pipeline from a slower, irregular source (e.g. low-fps screen share):
    playback is always steady in real time, a pose becomes visible the
    instant it's computed, and stays smoothly on screen until superseded —
    no artificial startup delay or catch-up burst needed.
    """

    # How long the "STREAM COMPLETE" banner is shown right after a session
    # ends, before ticking switches to labeling frames as a replay.
    COMPLETE_BANNER_SECONDS = 2.0

    # Used only for the "pending" placeholder pipeline the very first time
    # it's ever needed, before any real session has told us the true
    # width/height/fps. Overwritten as soon as a real session connects.
    DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FPS = 1280, 720, 20

    # Dark navy, deliberately NOT black: a viewer who saw the earlier
    # black-screen stride bug should be able to tell this apart at a glance
    # as an intentional "please wait" placeholder, not that bug again.
    PENDING_COLOR = (18, 22, 46)

    # If a run announces itself (notify_pending) but a real session never
    # follows — e.g. MDM sampling crashed before HUGS rendering ever
    # connected — say so instead of a "please wait" counter climbing
    # forever with nothing actually happening.
    PENDING_TIMEOUT_SECONDS = 180

    # Negative (darkening), not positive: HUGS renders on a white background
    # (bg_color='white'), which is already clipped at 255 — a *positive*
    # brightness bump is invisible there (only the avatar would visibly
    # flash). Darkening works against any background.
    FLASH_BRIGHTNESS = -0.25
    FLASH_TICKS = 2  # ~100ms at 20fps — long enough to register, short enough to not obscure content

    def __init__(self, mediamtx_url: str):
        self.mediamtx_url = mediamtx_url
        self._lock = threading.Lock()
        self.pipeline = None
        self.appsrc = None
        self.overlay = None
        self.balance = None
        self.width = self.height = self.fps = None
        self._fps_num = None
        self._frame_num = 0  # monotonic pts counter across the pipeline's lifetime
        self._clip: list = []  # last completed clip, replayed on loop when idle
        self._pending_clip: list = []  # frames collected during the current live session
        self._live = False
        self._live_count = 0
        self._total_frames = None
        self._session_ended_at = None
        self._display_frame = None  # most-recent frame to hold/repeat each tick
        self._display_label = None
        self._pending = False  # a new run announced itself but hasn't sent frame 1 yet
        self._pending_started_at = None
        self._flash_dirty = False  # a new live frame arrived since the last tick
        self._flash_ticks_left = 0
        self._dark_frame_cache: dict = {}
        threading.Thread(target=self._tick_loop, daemon=True).start()

    def set_pending(self) -> None:
        """Call the instant a new run starts (before MDM/rendering even
        begin) so viewers immediately see a "please wait" placeholder
        instead of the previous run's stale replay loop."""
        width = self.width or self.DEFAULT_WIDTH
        height = self.height or self.DEFAULT_HEIGHT
        fps = self.fps or self.DEFAULT_FPS
        self._ensure_pipeline(width, height, fps)
        with self._lock:
            self._pending = True
            self._live = False
        self._pending_started_at = time.monotonic()

    def start_live_session(self, width: int, height: int, fps: int, total_frames: int = None) -> None:
        self._ensure_pipeline(width, height, fps)
        with self._lock:
            self._pending = False
            self._pending_clip = []
            self._live = True
            self._live_count = 0
            self._total_frames = total_frames

    def push_live_frame(self, payload: bytes) -> None:
        with self._lock:
            self._pending_clip.append(payload)
            self._live_count += 1
            total = self._total_frames if self._total_frames else "?"
            self._display_frame = payload
            self._display_label = f"LIVE - rendering frame {self._live_count}/{total}"
            self._flash_dirty = True

    def end_live_session(self) -> None:
        with self._lock:
            self._live = False
            if self._pending_clip:
                self._clip = self._pending_clip
            self._pending_clip = []
        self._session_ended_at = time.monotonic()

    def _dark_frame(self, width: int, height: int) -> bytes:
        key = (width, height)
        cached = self._dark_frame_cache.get(key)
        if cached is None:
            r, g, b = self.PENDING_COLOR
            cached = bytes((r, g, b)) * (width * height)
            self._dark_frame_cache = {key: cached}  # only ever need the current size
        return cached

    def _ensure_pipeline(self, width: int, height: int, fps: int) -> None:
        with self._lock:
            if self.pipeline is not None and (width, height, fps) == (self.width, self.height, self.fps):
                return
            if self.pipeline is not None:
                print("[gst_stream_server] stream caps changed, rebuilding pipeline")
                self.pipeline.set_state(Gst.State.NULL)
            self.width, self.height, self.fps = width, height, fps
            self._fps_num = Gst.SECOND // fps
            self._clip = []
            self._frame_num = 0
            self._display_frame = None
            self.pipeline = build_webrtc_pipeline(width, height, fps, self.mediamtx_url)
            self.appsrc = self.pipeline.get_by_name("src")
            self.overlay = self.pipeline.get_by_name("overlay")
            self.balance = self.pipeline.get_by_name("balance")

            bus = self.pipeline.get_bus()
            bus.add_signal_watch()

            def on_message(_bus, message):
                if message.type == Gst.MessageType.ERROR:
                    err, debug = message.parse_error()
                    print(f"[gst_stream_server] pipeline error: {err} ({debug})")

            bus.connect("message", on_message)
            self.pipeline.set_state(Gst.State.PLAYING)

    def _tick_loop(self) -> None:
        clip_pos = 0
        while True:
            if self.appsrc is None or self.fps is None:
                time.sleep(0.2)
                continue
            tick_start = time.monotonic()

            if self._pending and not self._live:
                waited = time.monotonic() - self._pending_started_at
                if waited > self.PENDING_TIMEOUT_SECONDS:
                    label = f"NO SIGNAL - waited {waited:.0f}s, run may have failed (check its logs)"
                else:
                    label = f"RENDERING - preparing first frame... ({waited:.0f}s)"
                self._push(self._dark_frame(self.width, self.height), label, flash=False)
            elif self._live:
                with self._lock:
                    frame, label = self._display_frame, self._display_label
                    is_new = self._flash_dirty
                    self._flash_dirty = False
                if frame is not None:  # None only for an instant before frame 1 of a session lands
                    self._push(frame, label, flash=is_new)
            elif self._clip:
                clip_len = len(self._clip)
                clip_pos %= clip_len
                since_end = (
                    time.monotonic() - self._session_ended_at
                    if self._session_ended_at is not None else float("inf")
                )
                if since_end < self.COMPLETE_BANNER_SECONDS:
                    label = f"STREAM COMPLETE - {clip_len} frames"
                else:
                    label = f"REPLAY (idle) - frame {clip_pos + 1}/{clip_len}"
                self._push(self._clip[clip_pos], label, flash=False)
                clip_pos += 1

            elapsed = time.monotonic() - tick_start
            time.sleep(max(0.0, 1.0 / self.fps - elapsed))

    def _push(self, payload: bytes, label: str = None, flash: bool = False) -> None:
        if label is not None and self.overlay is not None:
            self.overlay.set_property("text", label)
        if flash:
            self._flash_ticks_left = self.FLASH_TICKS
        if self.balance is not None:
            brightness = self.FLASH_BRIGHTNESS if self._flash_ticks_left > 0 else 0.0
            self.balance.set_property("brightness", brightness)
            if self._flash_ticks_left > 0:
                self._flash_ticks_left -= 1
        buf = _wrap_frame(payload, self.width, self.height)
        buf.pts = self._frame_num * self._fps_num
        buf.duration = self._fps_num
        self._frame_num += 1
        self.appsrc.emit("push-buffer", buf)


def handle_webrtc_connection(conn: socket.socket, header: dict, feeder: WebrtcFeeder) -> None:
    width, height, fps = header["width"], header["height"], header["fps"]
    total_frames = header.get("total_frames")
    frame_size = width * height * 3

    print(f"[gst_stream_server] session start: {width}x{height}@{fps}fps -> {feeder.mediamtx_url} (webrtc)")
    feeder.start_live_session(width, height, fps, total_frames)

    frame_idx = 0
    try:
        while True:
            length_bytes = _recv_exact(conn, 4)
            if length_bytes is None:
                print("[gst_stream_server] client disconnected mid-stream")
                break
            (length,) = struct.unpack(">I", length_bytes)
            if length == 0:
                break  # EOS marker
            if length != frame_size:
                raise ValueError(f"expected {frame_size} bytes, got {length}")
            payload = _recv_exact(conn, length)
            if payload is None:
                break
            feeder.push_live_frame(payload)
            frame_idx += 1
    finally:
        feeder.end_live_session()
        conn.sendall(b"\x01")  # ack: fully flushed
        print(f"[gst_stream_server] session done: {frame_idx} frames -> now looping last clip until next session")


def _recv_exact(conn: socket.socket, n: int):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def _read_header(conn: socket.socket):
    header_bytes = b""
    while not header_bytes.endswith(b"\n"):
        chunk = conn.recv(1)
        if not chunk:
            return None
        header_bytes += chunk
    return json.loads(header_bytes.decode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9977)
    parser.add_argument(
        "--mode", choices=["hls", "webrtc"], default="hls",
        help="hls: write local .m3u8/.ts (default). "
             "webrtc: RTSP-push to a MediaMTX instance, which serves WebRTC to browsers and "
             "loops the last clip between sessions "
             "(start it first: tools/mediamtx/mediamtx tools/mediamtx/mediamtx.yml)",
    )
    parser.add_argument(
        "--mediamtx-url", default="rtsp://127.0.0.1:8554/hugs_stream",
        help="RTSP RECORD URL to push to when --mode webrtc (must match a path in mediamtx.yml)",
    )
    args = parser.parse_args()

    Gst.init(None)
    feeder = WebrtcFeeder(args.mediamtx_url) if args.mode == "webrtc" else None

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print(f"[gst_stream_server] listening on {args.host}:{args.port} (mode={args.mode})")

    while True:
        conn, addr = server.accept()
        print(f"[gst_stream_server] connection from {addr}")
        try:
            header = _read_header(conn)
            if header is None:
                print("[gst_stream_server] client disconnected before sending header")
                continue
            if header.get("pending"):
                # Fire-and-forget: a new run is starting, no frames follow.
                if feeder is not None:
                    print("[gst_stream_server] new run announced, showing placeholder")
                    feeder.set_pending()
                continue
            if args.mode == "webrtc":
                handle_webrtc_connection(conn, header, feeder)
            else:
                handle_hls_connection(conn, header)
        except Exception as e:
            print(f"[gst_stream_server] error: {e}")
        finally:
            conn.close()


if __name__ == "__main__":
    main()
