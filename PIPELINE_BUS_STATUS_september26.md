# Pipeline Bus & Profiling — Status

_Last updated: 2026-09-24_

## Why it's ~20 seconds faster now

The benchmark showed HUGS (step 5) was the slowest part. Digging into it, we found HUGS was always rendering **200 extra frames** after every run — a preview clip of the character spinning in two fixed poses. Nobody used this preview. The real output (the video and pictures we actually want) comes from a different, earlier part of step 5.

We turned this extra preview off. Since it's not rendered anymore, step 5 finishes about 20 seconds sooner, with no change to the actual output.

## What we were trying to do

Two things:

1. Make the pipeline pass data between **every** step using GStreamer, instead of only using it at the very end (for the live video stream). Before this, each step wrote a file to disk and the next step read that file.
2. Add timing and resource tracking to every step, so we can see how long each part takes, and how much CPU, memory, disk, and GPU it uses.

## The pipeline has 5 steps now

1. **Speech to text** — Whisper turns an audio recording into text. (Skipped if you type a prompt instead of speaking.)
2. **LLM input generator** — llama3.2 (via Ollama) cleans up the text into the form `"a person <does something>"`.
3. **Motion generator** — MDM turns the text into a 3D skeleton motion.
4. **Coordinate converter** — fits the skeleton to a proper body model (SMPL), then rotates it so it lines up correctly in HUGS's world.
5. **3DGS generator** — HUGS renders the motion into a 3D Gaussian Splat animation and produces the PNG frames and final MP4 video. (PLY files are off by default — we don't need them.)

We used to count this as 6 steps and have a separate "show the video" step at the end. We dropped that, because the video is already streamed live to the browser as it renders in step 5 — there's no separate step needed just to play it back.

## What's done

- Steps 3, 4, and 5 now hand data to each other through a GStreamer server (the "broker") instead of writing files to disk. This is on by default.
- A `--save-intermediate` flag brings back the old file-based behavior, for cases where you want to reuse a saved motion file later (like running the same motion through HUGS again with different settings).
- Each step is timed, and split into "read", "compute", and "write" time where possible.
- Each step also records CPU %, memory used, disk read/write, and GPU %/memory, sampled while it runs.
- All of this shows up in the terminal at the end of a run, and is saved to `benchmark_timing.json` and `benchmark_timing.csv` in the run's output folder.
- Fixed a bug where the speech-to-text and LLM steps weren't being timed at all (they ran before timing even started).
- Turned off the unused 200-frame preview render in step 5 (see above) — about 20 seconds faster per run, same output.

## What's been tested

- The GStreamer broker was started and tested directly: sending data in, and reading it back out, works correctly.
- Found and fixed a real bug: if a step tried to *read* data before another step had *sent* it yet, it would fail immediately instead of waiting. Now it waits properly.
- Ran the "coordinate converter" step for real (fit + rotate) using the broker, and checked the output numbers were correct.
- Ran the whole pipeline in "dry run" mode (prints what it *would* do, without actually doing it) — the steps, names, and file layout all look right.
- Ran the whole pipeline for real, end to end. The benchmark numbers showed HUGS rendering was the slowest step, and pointed us to the unused 200-frame preview render (see "Why it's ~20 seconds faster now" above).

## What's NOT done yet

- The `plot_pipeline_latency_v2.py` chart script still uses old hardcoded numbers instead of reading the new benchmark files. Not required, just a nice-to-have for later.

## How to try it yourself

```bash
cd ~/modular-3dgs-avatar-pipeline

# 1. Start the broker (only needs to be done once)
./scripts/start_streaming.sh

# 2. Run the pipeline
/home/sigma/anaconda3/envs/hugs/bin/python scripts/run_text2hugs.py \
  --prompt "a person waves" \
  --out_root ./output_text2hugs \
  --scene bike \
  --center --tz 1.0
```

Add `--save-intermediate` if you want the old-style files saved to disk too.
