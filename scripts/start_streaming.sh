#!/bin/bash
# Starts the two background processes the pipeline needs:
#   1. MediaMTX      - re-exposes our GStreamer feed as WebRTC to the browser
#      (only used for --stream-live viewing)
#   2. gst_stream_server.py (--mode webrtc) - the "broker": encodes/pushes
#      HUGS frames to MediaMTX for --stream-live, AND (as of the pipeline-bus
#      work) relays every stage's data hand-off (MDM motion -> SMPL params ->
#      rotated motion -> HUGS) by default -- see scripts/pipeline_bus.py.
#      run_text2hugs.py fails fast at startup if this isn't reachable, unless
#      run with --save-intermediate (which still uses files, not the bus).
#
# Safe to run more than once: if a process is already running, it's left
# alone rather than started again. Run this once before any
# scripts/run_text2hugs.py run (not just --stream-live ones).

set -e
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

mkdir -p logs

if pgrep -f "tools/mediamtx/mediamtx" > /dev/null; then
    echo "[start_streaming] MediaMTX already running"
else
    echo "[start_streaming] starting MediaMTX..."
    nohup ./tools/mediamtx/mediamtx tools/mediamtx/mediamtx.yml > logs/mediamtx.log 2>&1 &
    disown
    sleep 1
fi

if pgrep -f "scripts/gst_stream_server.py" > /dev/null; then
    echo "[start_streaming] gst_stream_server already running"
else
    echo "[start_streaming] starting gst_stream_server (webrtc mode)..."
    nohup /home/sigma/anaconda3/envs/gstreamer/bin/python -u scripts/gst_stream_server.py \
        --port 9977 --mode webrtc --mediamtx-url rtsp://127.0.0.1:8554/hugs_stream \
        > logs/gst_stream_server.log 2>&1 &
    disown
    sleep 1
fi

echo ""
echo "Ready. The pipeline bus is now up -- run_text2hugs.py will use it by"
echo "default for every run (add --save-intermediate to also write the"
echo "conventional rotated_npz/etc. files to disk)."
echo ""
echo "For live viewing, open this in a browser and leave the tab open:"
echo "  http://127.0.0.1:9080/hugs_stream"
echo ""
echo "Example run:"
echo "  /home/sigma/anaconda3/envs/hugs/bin/python scripts/run_text2hugs.py \\"
echo "    --prompt \"a person waving\" --out_root ./output_text2hugs --stream-live"
