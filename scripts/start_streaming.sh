#!/bin/bash
# Starts the two background processes the live-stream viewer needs:
#   1. MediaMTX      - re-exposes our GStreamer feed as WebRTC to the browser
#   2. gst_stream_server.py (--mode webrtc) - encodes HUGS frames and pushes
#      them to MediaMTX, looping the last clip between renders
#
# Safe to run more than once: if a process is already running, it's left
# alone rather than started again. Run this once before using
# scripts/run_text2hugs.py --stream-live.

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
echo "Ready. Open this in a browser and leave the tab open:"
echo "  http://127.0.0.1:8889/hugs_stream"
echo ""
echo "Then run a pipeline with --stream-live, e.g.:"
echo "  /home/sigma/anaconda3/envs/hugs/bin/python scripts/run_text2hugs.py \\"
echo "    --prompt \"a person waving\" --out_root ./output_text2hugs --stream-live"
