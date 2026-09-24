# Live Streaming (`--stream-live`)

Watch the HUGS avatar animate in a browser **while it renders**, instead of
waiting for the whole clip and opening the finished `.mp4`. This is
additional to the normal pipeline — `result.mp4` is still produced exactly
as before, streaming just gives you a live preview in parallel.

```
scripts/run_text2hugs.py (hugs env)
      │  renders one frame, pushes it immediately, renders the next...
      ▼
hugs/utils/gst_stream.py — thin TCP client, no GStreamer dependency
      │  raw RGB24 frames over a local socket
      ▼
scripts/gst_stream_server.py --mode webrtc (gstreamer env)
      │  encodes H.264, burns in a status overlay, RTSP-pushes to MediaMTX
      ▼
tools/mediamtx/mediamtx
      │  re-exposes the feed as WebRTC — no browser code, no signaling to write
      ▼
your browser: http://127.0.0.1:9080/hugs_stream
```

Two separate conda envs are involved because GStreamer/PyGObject only lives
in the `gstreamer` env, while HUGS itself only lives in the `hugs` env — see
`hugs/utils/gst_stream.py`'s docstring for why.

## Quick start

```bash
cd /home/sigma/modular-3dgs-avatar-pipeline

# 1. Start the two background servers (safe to re-run — skips what's already up)
./scripts/start_streaming.sh

# 2. Open in a browser, leave the tab open
#    http://127.0.0.1:9080/hugs_stream

# 3. Run any pipeline with --stream-live added
/home/sigma/anaconda3/envs/hugs/bin/python scripts/run_text2hugs.py \
  --prompt "a person waving" \
  --out_root ./output_text2hugs \
  --stream-live
```

That's it — re-run step 3 with a new `--prompt` any time; the same browser
tab picks up each new run automatically. The servers from step 1 can stay
up indefinitely; you only need to run `start_streaming.sh` again after a
reboot or if you notice the stream is dead (see Troubleshooting).

## What the on-screen status text means

A status line is burned into the top-left corner of the stream itself (via
GStreamer's `textoverlay`), so it shows up in any viewer — no custom
browser code needed:

| Text | Meaning |
|---|---|
| `RENDERING - preparing first frame... (Ns)` | dark navy screen — a run started (MDM/SMPL-extraction is in progress) but hasn't reached the HUGS render loop yet |
| `NO SIGNAL - waited Ns, run may have failed` | pending state timed out (>180s) with no real session — check the run's terminal/log |
| `LIVE - rendering frame N/Total` | actively streaming — the background also flashes darker for ~100ms every time a genuinely new rendered frame lands, so you can tell real progress from a held frame |
| `STREAM COMPLETE - N frames` | shown for ~2s right after a run finishes |
| `REPLAY (idle) - frame N/Total` | the run is done; looping the finished clip until the next run starts |

## Why it's implemented this way (context for future changes)

- **Persistent pipeline, not one-per-run**: `gst_stream_server.py --mode
  webrtc` builds its GStreamer pipeline once and keeps the RTSP publish to
  MediaMTX alive for the server's whole lifetime, looping the last clip
  between runs (`WebrtcFeeder` class). Tearing the publish down at the end
  of each run (the first version of this did) means MediaMTX shows "stream
  not found" the instant rendering finishes — anyone who opens the page a
  few seconds late misses it entirely.
- **Steady ticker, not push-on-arrival**: HUGS only renders ~2 frames/sec
  (real GPU time per frame), far below the stream's declared 20fps. A
  background thread ticks at a fixed 1/20s and always pushes the
  most-recently-known frame (holding/repeating it if nothing new has
  arrived). Pushing straight from the network as frames arrived made the
  encoded video's timestamps lie about real elapsed time and looked jerky.
- **Row-stride gotcha**: raw RGB frames sent over the wire are tightly
  packed (`width*3` bytes/row), but GStreamer's default assumption for a
  raw-video buffer with no explicit video meta is `width*3` rounded UP to a
  multiple of 4. For odd widths (this pipeline commonly renders at 1266px
  wide → 3798 vs. the assumed 3800) that mismatch silently desyncs every
  row and degrades the image to black. Fixed via
  `GstVideo.buffer_add_video_meta_full` in `_wrap_frame()` — don't remove
  it even though it looks like unnecessary boilerplate.
- **No `webrtcbin`**: this machine's `gstreamer` conda env has no `libnice`,
  so gst-plugins-bad's WebRTC plugin never registers. MediaMTX does the
  actual SDP/ICE/DTLS-SRTP work instead; GStreamer only needs to RTSP-push
  an H.264 elementary stream to it (`rtspclientsink`), no `webrtcbin`
  required. If a future environment gets a `libnice`-enabled GStreamer
  build, this whole RTSP hop could be replaced with a direct `webrtcbin`
  pipeline for lower latency.

## Troubleshooting

**Browser shows "stream not found" and never recovers**
Check the two servers are actually running: `pgrep -fl "mediamtx|gst_stream_server"`.
If neither shows up, run `./scripts/start_streaming.sh` again.

**Screen is stuck on "RENDERING - preparing first frame..." past ~3 minutes**
The label itself will switch to `NO SIGNAL - waited Ns, run may have failed`
once it passes 180s. Check the run's own terminal output / log file for the
actual error (MDM or SMPL-extraction most likely).

**A different resolution/scene causes a visible blip when the run starts**
Expected — `gst_stream_server.py` rebuilds its GStreamer pipeline whenever
width/height/fps change (e.g. switching from the placeholder's default
1280x720 to a scene's real render size), which briefly disconnects and
reconnects the RTSP publish. Self-heals in well under a second.

**Multiple people want to watch at once**
Just have them open the same URL — MediaMTX fans a WebRTC stream out to any
number of viewers natively, nothing extra to configure.

**Want other machines on the LAN to watch, not just this one**
Replace `127.0.0.1` with this machine's LAN IP in the viewer URL
(`http://<this-machine-IP>:9080/hugs_stream`). Note there's no
authentication on that endpoint — anyone on the same network segment could
also open it. Fine for a trusted home/office network; add MediaMTX's
built-in auth (or a firewall rule / VPN) before doing this on a network you
don't fully trust.

## Known limitations / ideas for later

- **Camera is fixed per scene.** The camera path for every frame is decided
  upfront by the scene config (`cfg_files/release/neuman/...`), not
  something a viewer can steer live. True free-camera control would need a
  browser → server channel (WebSocket/DataChannel) carrying the requested
  pose, and would change HUGS rendering from "render the whole planned clip"
  to "render whichever frame was just requested" — a materially bigger
  change, not attempted here.
- **No auto-restart.** `start_streaming.sh` starts plain background
  processes; a reboot or crash needs a human to re-run it. A systemd user
  service (or a supervisor script) would make both processes come back on
  their own — worth doing if this needs to "just always work" unattended.
- **Encoder isn't latency-tuned.** `nvh264enc` currently runs with default
  settings. Properties like `zerolatency=true` and a smaller `gop-size`
  would shave a bit more latency off the LIVE portion of the stream.

## File reference

| File | Runs in | Purpose |
|---|---|---|
| `hugs/utils/gst_stream.py` | `hugs` env | Stdlib-only TCP client — no GStreamer dependency. Sends raw frames + the `notify_pending()` fire-and-forget ping. |
| `scripts/gst_stream_server.py` | `gstreamer` env | Owns the actual GStreamer pipeline: encoding, the status overlay, the persistent RTSP publish, the idle-loop and flash logic (`WebrtcFeeder`). |
| `tools/mediamtx/mediamtx.yml` | — | MediaMTX config: RTSP-in on 8554, WebRTC-out on 9080. |
| `scripts/start_streaming.sh` | shell | Convenience launcher for both background servers. |
| `hugs/trainer/gs_trainer.py` | `hugs` env | The HUGS animate loop — where each frame is rendered and immediately handed to `FrameStreamClient.push_frame()`. |
