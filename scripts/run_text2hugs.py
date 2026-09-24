#!/usr/bin/env python3
"""
End-to-end pipeline: Speech/Text → LLM → MDM → Coordinate Convert → HUGS 3DGS Render

Workflow (5 stages -- see StageBenchmark / benchmark_timing.json for per-stage
timing, resource usage, and read/compute/write phase breakdowns):
  1. Speech-to-text: Whisper transcribes an audio file / mic / browser
     recording to text. Skipped (not run) for plain --prompt text input.
  2. LLM input generator: llama3.2 (via Ollama, see speech_io.py) normalizes
     whatever text stage 1 (or --prompt) produced into "a person <motion>",
     with optional --refine-prompt extra cleanup on top.
  3. Motion generator: MDM turns the text prompt into a 3D joint trajectory
     (skeleton), Y-up.
  4. Coordinate converter: two sub-steps that get MDM's output into the
     format/frame HUGS needs -- (a) SMPLify-3D fits SMPL pose parameters
     (global_orient/body_pose/betas) to the joint trajectory, then (b) the
     fitted root is rotated+translated (RX=+90°, RZ=+180°) into HUGS's Z-up
     coordinate system.
  5. 3DGS generator: HUGS renders the motion into per-frame Gaussian Splats,
     producing PNG frames + the final MP4 (PLY export is opt-in via
     --save_ply, off by default). When --stream-live is set, frames stream
     live via GStreamer as they render (hugs/utils/gst_stream.py) -- there's
     no separate playback/render-serving stage, since streaming already
     happens inline here rather than as a later step over the finished file.

By default, stages 3-5 hand data to each other through a GStreamer-based
pipeline bus (scripts/pipeline_bus.py + scripts/gst_stream_server.py)
instead of files on disk -- start it first with ./scripts/start_streaming.sh.
Pass --save-intermediate to also archive the intermediate npz files to their
conventional on-disk paths (needed e.g. for run_k_benchmark.py-style reuse).

Usage:
  python scripts/run_text2hugs.py \
    --prompt "a person jumps" \
    --scene bike \
    --out_root ./output_text2hugs \
    --center \
    --tz 1.0

  # Dry run (prints commands without executing)
  python scripts/run_text2hugs.py \
    --prompt "a person walks" \
    --out_root ./test \
    --dry_run
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Speech I/O (optional — requires openai-whisper, sounddevice, scipy, higgs-audio)
sys.path.insert(0, str(Path(__file__).parent))
# Repo root, so `from hugs.utils.gst_stream import ...` resolves — `hugs` is a
# namespace package (no __init__.py) only found via sys.path, not pip-installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from speech_io import browser_record_and_transcribe, record_and_transcribe, refine_prompt, normalize_prompt, speak_text
    _SPEECH_AVAILABLE = True
except ImportError:
    _SPEECH_AVAILABLE = False

# Pipeline bus (stage-to-stage data transport, default instead of disk files)
# and per-stage resource/phase profiling -- both in this same scripts/ dir.
import pipeline_bus
from pipeline_profiling import PhaseTimer, ResourceSampler, merge_phase_dicts, merge_resource_summaries

# Stage keys the bus uses to key each hand-off -- see scripts/pipeline_bus.py.
BUS_STAGE_MDM_RESULTS = "mdm_results"        # stage 3 (motion generator) -> stage 4 (coordinate converter)
BUS_STAGE_SMPL_EXTRACT = "smpl_extract"      # stage 4 internal hop: SMPL fit -> rotate
BUS_STAGE_ROTATED_MOTION = "rotated_motion"  # stage 4 (coordinate converter) -> stage 5 (3DGS generator)


# Default paths — override with --mdm_repo / --mdm_py if your layout differs
DEFAULT_MDM_REPO = Path.home() / "motion-diffusion-model"
DEFAULT_HUGS_REPO = Path(__file__).resolve().parent.parent   # this repo root
DEFAULT_MDM_PY = Path.home() / "anaconda3/envs/mdm/bin/python"
DEFAULT_HUGS_PY = Path(sys.executable)                       # current Python (run from hugs env)

class StageBenchmark:
    """Lightweight per-stage timer. Call start() before each stage, end() after."""

    def __init__(self) -> None:
        self.stages: List[dict] = []
        self._t0: Optional[float] = None
        self._num: Optional[int] = None
        self._name: Optional[str] = None

    def start(self, num: int, name: str) -> None:
        self._num = num
        self._name = name
        self._t0 = time.perf_counter()

    def end(
        self,
        status: str = 'success',   # 'success' | 'failed' | 'skipped'
        error: Optional[str] = None,
        output_path: Optional[str] = None,
        log_file: Optional[str] = None,
        phases: Optional[dict] = None,     # {'read': s, 'compute': s, 'write': s} or None
        resource: Optional[dict] = None,   # ResourceSampler.summary() or None
    ) -> dict:
        elapsed = round(time.perf_counter() - self._t0, 3) if self._t0 is not None else None
        record = {
            'stage':            self._num,
            'name':             self._name,
            'duration_seconds': elapsed,
            'status':           status,
            'error':            error,
            'output_path':      output_path,
            'log_file':         log_file,
            'end_iso':          datetime.now().isoformat(),
            'phases':           phases,
            'resource':         resource,
        }
        self.stages.append(record)
        self._t0 = None
        return record

    def total_seconds(self) -> float:
        return round(sum(s['duration_seconds'] or 0.0 for s in self.stages), 3)

    def save(self, run_dir: Path) -> tuple:
        """Write benchmark_timing.json and benchmark_timing.csv into run_dir."""
        import csv as _csv

        summary = {'total_duration_seconds': self.total_seconds(), 'stages': self.stages}

        json_path = run_dir / 'benchmark_timing.json'
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)

        csv_path = run_dir / 'benchmark_timing.csv'
        fields = [
            'stage', 'name', 'duration_seconds', 'status', 'error', 'output_path', 'log_file', 'end_iso',
            'phase_read_s', 'phase_compute_s', 'phase_write_s',
            'res_cpu_pct_avg', 'res_cpu_pct_max', 'res_rss_mb_max', 'res_mem_pct_avg',
            'res_io_read_mb', 'res_io_write_mb',
            'res_gpu_util_pct_avg', 'res_gpu_util_pct_max', 'res_gpu_mem_mb_max',
        ]
        rows = []
        for s in self.stages:
            row = dict(s)
            phases = row.pop('phases', None) or {}
            resource = row.pop('resource', None) or {}
            row['phase_read_s'] = phases.get('read')
            row['phase_compute_s'] = phases.get('compute')
            row['phase_write_s'] = phases.get('write')
            row['res_cpu_pct_avg'] = resource.get('cpu_pct_avg')
            row['res_cpu_pct_max'] = resource.get('cpu_pct_max')
            row['res_rss_mb_max'] = resource.get('rss_mb_max')
            row['res_mem_pct_avg'] = resource.get('mem_pct_avg')
            row['res_io_read_mb'] = resource.get('io_read_mb')
            row['res_io_write_mb'] = resource.get('io_write_mb')
            row['res_gpu_util_pct_avg'] = resource.get('gpu_util_pct_avg')
            row['res_gpu_util_pct_max'] = resource.get('gpu_util_pct_max')
            row['res_gpu_mem_mb_max'] = resource.get('gpu_mem_mb_max')
            rows.append(row)

        with open(csv_path, 'w', newline='') as f:
            w = _csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)

        return json_path, csv_path


# HUGS scene configurations
SCENE_CONFIGS = {
    "bike": {
        "human_ckpt": "output/pretrained_models/bike/human_final.pth",
        "scene_ckpt": "output/pretrained_models/bike/scene_final.pth",
    },
    "citron": {
        "human_ckpt": "output/pretrained_models/citron/human_final.pth",
        "scene_ckpt": "output/pretrained_models/citron/scene_final.pth",
    },
    "jogging": {
        "human_ckpt": "output/pretrained_models/jogging/human_final.pth",
        "scene_ckpt": "output/pretrained_models/jogging/scene_final.pth",
    },
    "lab": {
        "human_ckpt": "output/pretrained_models/lab/human_final.pth",
        "scene_ckpt": "output/pretrained_models/lab/scene_final.pth",
    },
    "parkinglot": {
        "human_ckpt": "output/pretrained_models/parkinglot/human_final.pth",
        "scene_ckpt": "output/pretrained_models/parkinglot/scene_final.pth",
    },
    "seattle": {
        "human_ckpt": "output/pretrained_models/seattle/human_final.pth",
        "scene_ckpt": "output/pretrained_models/seattle/scene_final.pth",
    },
}


def slugify(text: str, max_len: int = 30) -> str:
    """Convert text to filesystem-safe slug."""
    slug = re.sub(r'[^\w\s-]', '', text.lower())
    slug = re.sub(r'[-\s]+', '_', slug)
    return slug[:max_len].strip('_')


def run_command(
    cmd: List[str],
    description: str,
    log_file: Optional[Path] = None,
    cwd: Optional[Path] = None,
    dry_run: bool = False,
    sampler: Optional[ResourceSampler] = None,
) -> bool:
    """Execute a command and log output. If `sampler` is given, it's start()ed
    against the child's pid right after launch and stop()ed once it exits, so
    CPU/mem/disk/GPU usage gets attributed to this specific stage -- this is
    why the command runs via Popen+wait() rather than a single blocking
    subprocess.run() call."""
    print(f"\n{'='*80}")
    print(f"Step: {description}")
    print(f"Command: {' '.join(str(c) for c in cmd)}")
    if cwd:
        print(f"Working dir: {cwd}")
    if log_file:
        print(f"Log: {log_file}")
    print(f"{'='*80}")

    if dry_run:
        print("[DRY RUN] Command not executed")
        return True

    log_fh = None
    try:
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_fh = open(log_file, 'w')
            proc = subprocess.Popen(cmd, cwd=cwd, stdout=log_fh, stderr=subprocess.STDOUT)
        else:
            proc = subprocess.Popen(cmd, cwd=cwd)

        if sampler is not None:
            sampler.start(proc.pid)

        returncode = proc.wait()

        if sampler is not None:
            sampler.stop()

        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, cmd)

        print(f"✓ Command succeeded")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        if log_file and log_file.exists():
            print(f"See log file: {log_file}")
        return False
    finally:
        if log_fh is not None:
            log_fh.close()


def find_file(root: Path, filename: str) -> Optional[Path]:
    """Recursively find a file by name under root."""
    matches = list(root.rglob(filename))
    if matches:
        return matches[0]  # Return first match
    return None


def find_newest_mp4(search_root: Path, after_time: Optional[float] = None) -> Optional[Path]:
    """Find the most recent .mp4 file (optionally created after a timestamp).
    
    Priority order:
      1. anim_*.mp4  — actual custom-motion animation render
      2. Any other *.mp4 (newest by mtime)
    """
    mp4_files = list(search_root.rglob("*.mp4"))

    if after_time:
        mp4_files = [f for f in mp4_files if f.stat().st_mtime > after_time]

    if not mp4_files:
        return None

    # Prefer anim_*.mp4 (the actual animated output) over canonical pose videos
    anim_files = [f for f in mp4_files if f.name.startswith("anim_")]
    if anim_files:
        return max(anim_files, key=lambda p: p.stat().st_mtime)

    # Fallback: return newest by modification time
    return max(mp4_files, key=lambda p: p.stat().st_mtime)


def build_mdm_cmd(
    prompt: str,
    out_dir: Path,
    mdm_repo: Path,
    mdm_py: Path,
    seed: int,
    steps: int,
) -> List[str]:
    """
    Build MDM sampling command.
    
    MDM automatically creates output in:
    save/humanml_enc_512_50steps/samples_humanml_enc_512_50steps_000750000_seed{seed}_{prompt}/
    
    The output includes hugs_smpl_original.npz which is ready for HUGS rendering.
    """
    cmd = [
        str(mdm_py),
        "-m", "sample.generate",
        "--model_path", str(mdm_repo / "save/humanml_enc_512_50steps/model000750000.pt"),
        "--text_prompt", prompt,
        "--num_samples", "1",
        "--num_repetitions", "1",
        "--guidance_param", "2.5",
        "--seed", str(seed),
    ]
    
    return cmd



def main():
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: Text → MDM → HUGS render",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/run_text2hugs.py \\
    --prompt "a person jumps" \\
    --out_root ./output_text2hugs

  # With custom scene and translation
  python scripts/run_text2hugs.py \\
    --prompt "a person walks forward" \\
    --scene citron \\
    --out_root ./output \\
    --center \\
    --tz 1.5

  # Dry run to test commands
  python scripts/run_text2hugs.py \\
    --prompt "test" \\
    --out_root ./test \\
    --dry_run
        """,
    )
    
    # Required arguments
    parser.add_argument(
        "--prompt",
        default=None,
        help="Text prompt for MDM motion generation. Omit when using --speech-input.",
    )
    parser.add_argument(
        "--out_root",
        required=True,
        type=Path,
        help="Root output directory for all pipeline artifacts",
    )
    
    # Scene and rendering
    parser.add_argument(
        "--scene",
        default="bike",
        choices=list(SCENE_CONFIGS.keys()),
        help="HUGS scene to render (default: bike)",
    )
    
    # Repository and Python paths
    parser.add_argument(
        "--mdm_repo",
        type=Path,
        default=DEFAULT_MDM_REPO,
        help=f"Path to MDM repository (default: {DEFAULT_MDM_REPO})",
    )
    parser.add_argument(
        "--hugs_repo",
        type=Path,
        default=DEFAULT_HUGS_REPO,
        help=f"Path to HUGS repository (default: {DEFAULT_HUGS_REPO})",
    )
    parser.add_argument(
        "--mdm_py",
        type=Path,
        default=DEFAULT_MDM_PY,
        help=f"Path to MDM Python executable (default: {DEFAULT_MDM_PY})",
    )
    parser.add_argument(
        "--hugs_py",
        type=Path,
        default=DEFAULT_HUGS_PY,
        help=f"Path to HUGS Python executable (default: {DEFAULT_HUGS_PY})",
    )
    
    # MDM parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="Random seed for MDM (default: 10)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of diffusion steps for MDM (default: 50)",
    )
    
    # Rotation and translation
    parser.add_argument(
        "--tx",
        type=float,
        default=0.0,
        help="Horizontal X offset after centering (default: 0.0)",
    )
    parser.add_argument(
        "--ty",
        type=float,
        default=0.0,
        help="Vertical Y offset (up/down) after centering (default: 0.0). Use positive values to lift avatar up.",
    )
    parser.add_argument(
        "--tz",
        type=float,
        default=1.0,
        help="Depth Z offset after centering (default: 1.0). Controls how far in front of camera.",
    )
    parser.add_argument(
        "--ground",
        type=float,
        default=1.0,
        help="Snap lowest Z frame to this value to fix floating avatar (default: 1.0, set to None to disable)",
    )
    
    parser.add_argument(
        "--center",
        action="store_true",
        help="Center translation to mean 0 before rendering",
    )

    # Rendering mode
    parser.add_argument(
        "--bg_color",
        default="white",
        choices=["white", "black"],
        help="Background color for human-only rendering (default: white)",
    )
    parser.add_argument(
        "--save_ply",
        action="store_true",
        help="Save per-frame Gaussian Splat .ply files during animation into anim_ply/ folder",
    )
    parser.add_argument(
        "--stream-live",
        action="store_true",
        help="Push each rendered frame to a live GStreamer/HLS server as it's rendered, "
             "instead of waiting for the whole clip before producing a video "
             "(start scripts/gst_stream_server.py first, in the 'gstreamer' conda env)",
    )
    parser.add_argument(
        "--stream-host",
        default="127.0.0.1",
        help="Host where scripts/gst_stream_server.py is listening (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--stream-port",
        type=int,
        default=9977,
        help="Port where scripts/gst_stream_server.py is listening (default: 9977)",
    )
    parser.add_argument(
        "--stream-segment-duration",
        type=float,
        default=1.0,
        help="Seconds per HLS segment for --stream-live (default: 1.0)",
    )
    parser.add_argument(
        "--subsample-k",
        type=int,
        default=1,
        choices=[1, 2, 4],
        dest="subsample_k",
        help="Export every k-th frame during animation (1=all frames, 2=half, 4=quarter; default: 1)",
    )
    parser.add_argument(
        "--orbit-camera",
        action="store_true",
        dest="orbit_camera",
        help="Circle the camera 360° around the avatar over the course of the clip, instead of the "
             "default per-scene fixed/sliding camera that always faces one direction.",
    )
    parser.add_argument(
        "--orbit-dist",
        type=float,
        default=3.0,
        dest="orbit_dist",
        help="Distance (world units) from the avatar to the orbiting camera, only used with "
             "--orbit-camera (default: 3.0; tuned for --scene bike, may need adjusting for other scenes)",
    )

    # Speech I/O
    parser.add_argument(
        "--audio-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to a pre-recorded audio file (wav/mp3/ogg/webm/etc.) to transcribe with Whisper and use as the prompt",
    )
    parser.add_argument(
        "--speech-input",
        action="store_true",
        help="Record from the local ALSA mic and use Whisper transcription as the prompt",
    )
    parser.add_argument(
        "--browser-input",
        action="store_true",
        help=(
            "Serve a recording page over HTTP so you can speak from a remote browser "
            "(use this when working via AnyDesk/VNC/SSH where the local mic is unavailable)"
        ),
    )
    parser.add_argument(
        "--browser-port",
        type=int,
        default=9876,
        help="Port for the --browser-input HTTP server (default: 9876)",
    )
    parser.add_argument(
        "--speech-output",
        action="store_true",
        help="Read the transcribed prompt and final status aloud (HiggsAudio v2, falls back to espeak)",
    )
    parser.add_argument(
        "--tts-save-wav",
        type=Path,
        default=None,
        metavar="PATH",
        help="Save TTS speech to this WAV file instead of (or in addition to) playing — useful over AnyDesk/VNC where audio forwarding may not work",
    )
    parser.add_argument(
        "--record-duration",
        type=float,
        default=8.0,
        metavar="SECS",
        help="Microphone recording duration in seconds for --speech-input (default: 8.0)",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size for STT (default: base)",
    )
    parser.add_argument(
        "--refine-prompt",
        action="store_true",
        help="Use a local Ollama LLM to rewrite the Whisper transcription into a clean MDM motion prompt",
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3.2",
        help="Ollama model to use for prompt refinement (default: llama3.2)",
    )
    parser.add_argument(
        "--alsa-device",
        default="plughw:0,0",
        help="ALSA capture device for arecord fallback (default: plughw:0,0). "
             "Run 'arecord -l' to list cards. E.g. 'plughw:CARD=PCH,DEV=0'.",
    )
    parser.add_argument(
        "--higgs-model-path",
        default="bosonai/higgs-audio-v2-generation-3B-base",
        help="HiggsAudio v2 model path or HuggingFace ID (default: bosonai/higgs-audio-v2-generation-3B-base)",
    )
    parser.add_argument(
        "--higgs-tokenizer-path",
        default="bosonai/higgs-audio-v2-tokenizer",
        help="HiggsAudio v2 tokenizer path or HuggingFace ID (default: bosonai/higgs-audio-v2-tokenizer)",
    )

    # Execution control
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        dest="save_intermediate",
        help="Also write stage 4's (coordinate converter) sub-step outputs (hugs_smpl_original.npz, "
             "hugs_smpl_upright.npz) to their conventional on-disk paths, in addition to the default "
             "pipeline-bus transport. Needed for workflows that re-point HUGS at a saved rotated_npz "
             "from a past run (e.g. a subsample-k sweep) without re-running MDM/coordinate conversion.",
    )
    
    args = parser.parse_args()

    start_time = datetime.now()
    executed_commands = []
    bench = StageBenchmark()

    # ====================
    # Stage 1: Speech-to-text -- Whisper transcribes an audio file / mic /
    # browser recording to raw text. Not run (skipped) for plain --prompt
    # text input.
    # ====================
    bench.start(1, "Speech to text")
    print(f"\n[1/5] Speech-to-text...")
    stt_source = None
    if args.audio_file:
        if not _SPEECH_AVAILABLE:
            print("❌ --audio-file requires openai-whisper.")
            print("   pip install openai-whisper")
            sys.exit(1)
        audio_path = args.audio_file.resolve()
        if not audio_path.exists():
            print(f"❌ Audio file not found: {audio_path}")
            sys.exit(1)
        from speech_io import transcribe
        args.prompt = transcribe(audio_path, model_size=args.whisper_model)
        if not args.prompt:
            print("❌ Whisper returned empty transcription. Please try again.")
            sys.exit(1)
        stt_source = f"file:{audio_path.name}"
    elif args.browser_input:
        if not _SPEECH_AVAILABLE:
            print("❌ --browser-input requires openai-whisper.")
            print("   pip install openai-whisper")
            sys.exit(1)
        args.prompt = browser_record_and_transcribe(
            port=args.browser_port,
            model_size=args.whisper_model,
        )
        if not args.prompt:
            print("❌ Whisper returned empty transcription. Please try again.")
            sys.exit(1)
        stt_source = "browser"
    elif args.speech_input:
        if not _SPEECH_AVAILABLE:
            print("❌ --speech-input requires openai-whisper and sounddevice.")
            print("   pip install openai-whisper sounddevice scipy")
            sys.exit(1)
        args.prompt = record_and_transcribe(
            duration=args.record_duration,
            model_size=args.whisper_model,
            alsa_device=args.alsa_device,
        )
        if not args.prompt:
            print("❌ Whisper returned empty transcription. Please try again.")
            sys.exit(1)
        stt_source = "mic"
    elif args.prompt is None:
        print("❌ Either --prompt TEXT or --speech-input is required.")
        parser.print_usage()
        sys.exit(1)

    if stt_source:
        print(f"✓ Transcribed ({stt_source}): {args.prompt!r}")
    else:
        print("– Skipped (--prompt given directly, no audio input)")
    bench.end(
        status='success' if stt_source else 'skipped',
        error=None if stt_source else 'no audio input (--prompt given directly)',
        output_path=stt_source,
    )

    # ====================
    # Stage 2: LLM input generator -- llama3.2 (via Ollama, see speech_io.py)
    # normalizes whatever text stage 1 (or --prompt) produced into the
    # canonical "a person <motion description>" form MDM expects, plus
    # optional --refine-prompt extra cleanup on top.
    # ====================
    bench.start(2, "LLM input generator")
    print(f"\n[2/5] LLM input generator...")
    if _SPEECH_AVAILABLE:
        args.prompt = normalize_prompt(args.prompt, model=args.ollama_model)
        if args.refine_prompt:
            args.prompt = refine_prompt(args.prompt, model=args.ollama_model)
        print(f"✓ Normalized prompt: {args.prompt!r}")
        bench.end(status='success', output_path=args.prompt)
    else:
        print("– Skipped (speech_io not importable, prompt used as-is)")
        bench.end(status='skipped', error='speech_io not importable (normalize_prompt unavailable)')

    def _speak(text: str, out_wav: Optional[str] = None) -> None:
        """Speak text if --speech-output is enabled."""
        if args.speech_output:
            if not _SPEECH_AVAILABLE:
                print("[TTS] --speech-output requested but speech_io not importable. Skipping.")
                return
            speak_text(
                text,
                model_path=args.higgs_model_path,
                tokenizer_path=args.higgs_tokenizer_path,
                out_wav=out_wav,
            )

    # Validate paths
    if not args.dry_run:
        if not args.mdm_repo.exists():
            print(f"❌ MDM repository not found: {args.mdm_repo}")
            sys.exit(1)
        if not args.hugs_repo.exists():
            print(f"❌ HUGS repository not found: {args.hugs_repo}")
            sys.exit(1)
        if not args.mdm_py.exists():
            print(f"❌ MDM Python not found: {args.mdm_py}")
            sys.exit(1)
        if not args.hugs_py.exists():
            print(f"❌ HUGS Python not found: {args.hugs_py}")
            sys.exit(1)
        if not pipeline_bus.is_broker_alive(host=args.stream_host, port=args.stream_port):
            print(f"❌ Pipeline bus/broker not reachable at {args.stream_host}:{args.stream_port}")
            print("   Stages 3-5 hand off data through it by default (no --save-intermediate).")
            print("   Start it first:  ./scripts/start_streaming.sh")
            sys.exit(1)

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = slugify(args.prompt)
    run_id = f"{timestamp}_{slug}"  # also the pipeline-bus key for this run's stage hand-offs
    run_dir = args.out_root / run_id
    
    # Create subdirectories
    mdm_out_dir = run_dir / "mdm_out"
    smpl_npz_dir = run_dir / "smpl_npz"
    rotated_npz_dir = run_dir / "rotated_npz"
    hugs_logs_dir = run_dir / "hugs_logs"
    final_dir = run_dir / "final"
    
    for d in [mdm_out_dir, smpl_npz_dir, rotated_npz_dir, hugs_logs_dir, final_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("Speech/Text → MDM → HUGS Pipeline")
    print(f"{'='*80}")
    print(f"Prompt:     {args.prompt}")
    _speak(f"{args.prompt}")
    print(f"Scene:      {args.scene}")
    print(f"Run dir:    {run_dir}")
    print(f"Seed:       {args.seed}")
    print(f"Steps:      {args.steps}")
    print(f"Center:     {args.center}")
    print(f"TZ offset:  {args.tz}")
    print(f"Bg color:   {args.bg_color}")
    print(f"Dry run:    {args.dry_run}")
    print(f"{'='*80}\n")

    if args.stream_live and not args.dry_run:
        # Tell the stream server we're starting now, well before HUGS
        # rendering (stage 5) actually connects with real frames — so
        # viewers see a "please wait, rendering..." placeholder for the
        # ~motion-generation + coordinate-conversion time too, instead of the
        # previous run's stale replay loop or nothing.
        from hugs.utils.gst_stream import notify_pending
        notify_pending(host=args.stream_host, port=args.stream_port)

    # ====================
    # Stage 3: Motion generator -- MDM turns the text prompt into a 3D joint
    # trajectory (skeleton) via motion diffusion, Y-up.
    # ====================
    bench.start(3, "Motion generator")
    print(f"\n[3/5] Motion generator (MDM)...")
    mdm_cmd = build_mdm_cmd(
        prompt=args.prompt,
        out_dir=mdm_out_dir,
        mdm_repo=args.mdm_repo,
        mdm_py=args.mdm_py,
        seed=args.seed,
        steps=args.steps,
    )
    
    mdm_log = mdm_out_dir / "mdm.log"
    mdm_sampler = ResourceSampler()
    if not run_command(
        mdm_cmd,
        "Generate motion with MDM",
        log_file=mdm_log,
        cwd=args.mdm_repo,
        dry_run=args.dry_run,
        sampler=mdm_sampler,
    ):
        bench.end(status='failed', log_file=str(mdm_log), resource=mdm_sampler.summary())
        print("❌ MDM generation failed")
        sys.exit(1)

    executed_commands.append({
        "stage": "mdm",
        "cmd": " ".join(str(c) for c in mdm_cmd),
        "cwd": str(args.mdm_repo),
    })
    bench.end(
        status='skipped' if args.dry_run else 'success',
        output_path=str(mdm_out_dir),
        log_file=str(mdm_log),
        resource=mdm_sampler.summary(),
    )

    # ====================
    # Stage 4: Coordinate converter -- MDM's output is a Y-up 3D joint
    # trajectory, not the SMPL pose parameters + Z-up frame HUGS needs. Two
    # sub-steps, both via the pipeline bus by default (--save-intermediate
    # also archives each sub-step's npz to disk):
    #   a) SMPLify-3D fits SMPL pose params (global_orient/body_pose/betas)
    #      to MDM's joint trajectory (extract_smpl_params.py)
    #   b) rotate+translate the fitted root by RX=+90°, RZ=+180° into HUGS's
    #      coordinate system (rotate_hugs_motion_v2.py)
    # Both sub-steps' phase/resource stats are combined into this one stage's
    # bench entry via merge_phase_dicts/merge_resource_summaries.
    # ====================
    bench.start(4, "Coordinate converter")
    print(f"\n[4/5] Coordinate converter (SMPL fit + rotate)...")

    target_npz = smpl_npz_dir / "hugs_smpl_original.npz"
    extract_log = mdm_out_dir / "extract_smpl.log"
    extract_phases_path = mdm_out_dir / "extract_smpl.phases.json"
    rotated_npz = rotated_npz_dir / "hugs_smpl_upright.npz"
    rotate_log = rotated_npz_dir / "rotate.log"
    rotate_phases_path = rotated_npz_dir / "rotate.phases.json"
    extract_sampler = ResourceSampler()
    rotate_sampler = ResourceSampler()

    if not args.dry_run:
        # Locate the latest MDM samples directory
        mdm_save_dir = args.mdm_repo / "save" / "humanml_enc_512_50steps"
        latest_sample_dir = None

        if mdm_save_dir.exists():
            samples_dirs = sorted(
                mdm_save_dir.glob("samples_*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if samples_dirs:
                latest_sample_dir = samples_dirs[0]
                print(f"Found latest MDM output dir: {latest_sample_dir.name}")

        # Find results.npy
        results_npy = None
        if latest_sample_dir and (latest_sample_dir / "results.npy").exists():
            results_npy = latest_sample_dir / "results.npy"
        else:
            results_npy = find_file(args.mdm_repo / "save", "results.npy")

        if results_npy is None or not results_npy.exists():
            bench.end(status='failed', error='results.npy not found', log_file=str(mdm_log))
            print("❌ results.npy not found in MDM save directory")
            print(f"Searched in: {mdm_save_dir}")
            print(f"MDM log: {mdm_log}")
            sys.exit(1)

        print(f"✓ Found MDM results: {results_npy}")

        # results.npy itself always exists on disk (MDM's own vendored save
        # convention -- not something we can avoid without forking MDM), but
        # this stage no longer reads that path directly by default: hand it
        # off via the pipeline bus instead, same as every later hop.
        pipeline_bus.push_file(run_id, BUS_STAGE_MDM_RESULTS, results_npy,
                                host=args.stream_host, port=args.stream_port)

        # -- a) SMPLify-3D fit: results.npy → hugs_smpl_original.npz --------
        extract_script = args.mdm_repo / "sample" / "extract_smpl_params.py"
        extract_cmd = [
            str(args.mdm_py),
            str(extract_script),
            "--bus-pull", BUS_STAGE_MDM_RESULTS,
            "--bus-push", BUS_STAGE_SMPL_EXTRACT,
            "--bus-run-id", run_id,
            "--bus-host", args.stream_host,
            "--bus-port", str(args.stream_port),
            "--pipeline-bus-path", str(args.hugs_repo / "scripts"),
            "--phase-timing-out", str(extract_phases_path),
        ]
        if args.save_intermediate:
            # Use absolute path so it is not resolved relative to mdm_repo cwd
            extract_cmd.extend(["--output", str(target_npz.resolve())])

        if not run_command(
            extract_cmd,
            "Coordinate converter, step a: SMPLify-3D fit (results.npy → hugs_smpl_original.npz, via pipeline bus)",
            log_file=extract_log,
            cwd=args.mdm_repo,   # must run from MDM root for relative model paths
            dry_run=args.dry_run,
            sampler=extract_sampler,
        ):
            bench.end(
                status='failed', log_file=str(extract_log),
                phases=PhaseTimer.load(extract_phases_path),
                resource=extract_sampler.summary(),
            )
            print("❌ SMPL fit failed")
            print(f"See log: {extract_log}")
            sys.exit(1)

        executed_commands.append({
            "stage": "extract_smpl",
            "cmd": " ".join(str(c) for c in extract_cmd),
            "cwd": str(args.mdm_repo),
        })

        if args.save_intermediate and not target_npz.exists():
            raise FileNotFoundError(
                f"hugs_smpl_original.npz not produced by extract_smpl_params.py\n"
                f"Expected: {target_npz}\n"
                f"Check log: {extract_log}"
            )

        print(f"✓ SMPL params extracted (bus stage: {BUS_STAGE_SMPL_EXTRACT})"
              + (f", archived to: {target_npz}" if args.save_intermediate else ""))

        # -- b) rotate+translate into HUGS coordinates -----------------------
        rotate_script = args.hugs_repo / "scripts/rotate_hugs_motion_v2.py"
        rotate_cmd = [
            str(args.hugs_py),
            str(rotate_script),
            "--bus-pull", BUS_STAGE_SMPL_EXTRACT,
            "--bus-push", BUS_STAGE_ROTATED_MOTION,
            "--bus-run-id", run_id,
            "--bus-host", args.stream_host,
            "--bus-port", str(args.stream_port),
            "--phase-timing-out", str(rotate_phases_path),
            "--rx", "90",
            "--rz", "180",
        ]
        if args.save_intermediate:
            rotate_cmd.extend(["--output", str(rotated_npz)])

        if args.center:
            rotate_cmd.append("--center")

        rotate_cmd.extend(["--tx", str(args.tx)])
        rotate_cmd.extend(["--ty", str(args.ty)])
        rotate_cmd.extend(["--tz", str(args.tz)])

        if args.ground is not None:
            rotate_cmd.extend(["--ground", str(args.ground)])

        if not run_command(
            rotate_cmd,
            "Coordinate converter, step b: rotate to HUGS coords (RX=+90°, RZ=+180°, via pipeline bus)",
            log_file=rotate_log,
            cwd=args.hugs_repo,
            dry_run=args.dry_run,
            sampler=rotate_sampler,
        ):
            bench.end(
                status='failed', log_file=str(rotate_log),
                phases=merge_phase_dicts([PhaseTimer.load(extract_phases_path), PhaseTimer.load(rotate_phases_path)]),
                resource=merge_resource_summaries([extract_sampler.summary(), rotate_sampler.summary()]),
            )
            print("❌ Rotation failed")
            sys.exit(1)

        executed_commands.append({
            "stage": "rotate",
            "cmd": " ".join(str(c) for c in rotate_cmd),
            "cwd": str(args.hugs_repo),
        })
        print(f"✓ Rotated to HUGS coordinates (bus stage: {BUS_STAGE_ROTATED_MOTION})"
              + (f", archived to: {rotated_npz}" if args.save_intermediate else ""))

        bench.end(
            status='success',
            output_path=str(rotated_npz) if args.save_intermediate else None,
            log_file=f"{extract_log}, {rotate_log}",
            phases=merge_phase_dicts([PhaseTimer.load(extract_phases_path), PhaseTimer.load(rotate_phases_path)]),
            resource=merge_resource_summaries([extract_sampler.summary(), rotate_sampler.summary()]),
        )

    else:
        print(f"[DRY RUN] Would run extract_smpl_params.py + rotate_hugs_motion_v2.py (via pipeline bus)")
        if args.save_intermediate:
            print(f"[DRY RUN] Would also archive to: {target_npz}, {rotated_npz}")
        bench.end(status='skipped')

    # ====================
    # Stage 5: 3DGS generator -- HUGS renders the (Z-up, SMPL-parameterized)
    # motion into per-frame Gaussian Splats, producing PNG frames + the final
    # MP4 (PLY export is opt-in via --save_ply, off by default). Frames
    # stream live via GStreamer as they render when --stream-live is set
    # (hugs/utils/gst_stream.py) -- there's no separate playback/render-
    # serving stage, since that streaming already happens inline here rather
    # than as a later step over the finished file.
    # ====================
    bench.start(5, "3DGS generator")
    print(f"\n[5/5] 3DGS generator (HUGS render)...")

    scene_cfg = SCENE_CONFIGS[args.scene]

    hugs_config = args.hugs_repo / "cfg_files/release/neuman/hugs_human_scene.yaml"
    hugs_phases_path = hugs_logs_dir / "hugs_phases.json"
    hugs_cmd = [
        str(args.hugs_py),
        "main.py",
        "--cfg_file", str(hugs_config),
        f"dataset.seq={args.scene}",
        "eval=true",
        "mode=human",
        f"bg_color={args.bg_color}",
        f"human.ckpt={scene_cfg['human_ckpt']}",
        f"save_anim_ply={'true' if args.save_ply else 'false'}",
        f"anim_subsample_k={args.subsample_k}",
        f"stream_live={'true' if args.stream_live else 'false'}",
        f"stream_host={args.stream_host}",
        f"stream_port={args.stream_port}",
        f"stream_segment_duration={args.stream_segment_duration}",
        f"phase_timing_out={hugs_phases_path}",
        # run_text2hugs.py never uses the canonical a_pose/da_pose preview
        # (final/ only gets anim_*.mp4 + anim_ply/) -- skip rendering it.
        "skip_canonical=true",
    ]
    if args.orbit_camera:
        hugs_cmd.append("orbit_camera=true")
        hugs_cmd.append(f"orbit_dist={args.orbit_dist}")
    if args.save_intermediate:
        hugs_cmd.append(f"custom_motion_path={rotated_npz}")
    else:
        # Default: HUGS pulls the rotated motion from the pipeline bus itself
        # (hugs/datasets/neuman.py) instead of reading a file path.
        hugs_cmd.append(f"custom_motion_bus_stage={BUS_STAGE_ROTATED_MOTION}")
        hugs_cmd.append(f"custom_motion_bus_run_id={run_id}")

    hugs_log = hugs_logs_dir / "hugs.log"

    # Record timestamp before HUGS run to find new mp4 files
    before_hugs = datetime.now().timestamp()

    hugs_sampler = ResourceSampler()
    if not run_command(
        hugs_cmd,
        f"Render HUGS animation (scene={args.scene})",
        log_file=hugs_log,
        cwd=args.hugs_repo,
        dry_run=args.dry_run,
        sampler=hugs_sampler,
    ):
        bench.end(status='failed', log_file=str(hugs_log), resource=hugs_sampler.summary())
        print("❌ HUGS rendering failed")
        sys.exit(1)

    executed_commands.append({
        "stage": "hugs",
        "cmd": " ".join(str(c) for c in hugs_cmd),
        "cwd": str(args.hugs_repo),
    })

    # Collecting the finished mp4/PLY into final/ is bookkeeping on top of
    # the render above, not a separate stage -- it stays part of this same
    # bench(5) entry rather than getting its own stage number.
    if not args.dry_run:
        hugs_output_dir = args.hugs_repo / "output"
        newest_mp4 = find_newest_mp4(hugs_output_dir, after_time=before_hugs)
        
        if newest_mp4:
            final_mp4 = final_dir / "result.mp4"
            shutil.copy2(newest_mp4, final_mp4)
            print(f"✓ Video saved to: {final_mp4}")
            print(f"  Source: {newest_mp4}")
        else:
            print("⚠ No new mp4 files found after HUGS rendering")
            print(f"Searched in: {hugs_output_dir}")
            final_mp4 = None

        # Copy per-frame posed PLY files to the run's final directory
        anim_ply_dirs = sorted(
            hugs_output_dir.rglob("anim_ply"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        # Only consider dirs created after this pipeline run started
        anim_ply_dirs = [d for d in anim_ply_dirs if d.stat().st_mtime > before_hugs]
        final_ply_dir = None
        if anim_ply_dirs:
            src_ply_dir = anim_ply_dirs[0]
            final_ply_dir = final_dir / "anim_ply"
            if final_ply_dir.exists():
                shutil.rmtree(final_ply_dir)
            shutil.copytree(src_ply_dir, final_ply_dir)
            n_ply = len(list(final_ply_dir.glob("*.ply")))
            print(f"✓ {n_ply} per-frame posed PLY files saved to: {final_ply_dir}")
            print(f"  Source: {src_ply_dir}")
        else:
            print("⚠ No anim_ply directory found after HUGS rendering")
            final_ply_dir = None
    else:
        print(f"[DRY RUN] Would find newest mp4 in {args.hugs_repo / 'output'}")
        print(f"[DRY RUN] Would copy anim_ply/ posed PLY frames to {final_dir / 'anim_ply'}")
        final_mp4 = final_dir / "result.mp4"
        final_ply_dir = final_dir / "anim_ply"
    
    bench.end(
        status='skipped' if args.dry_run else 'success',
        output_path=str(final_dir),
        log_file=str(hugs_log),
        phases=PhaseTimer.load(hugs_phases_path),
        resource=hugs_sampler.summary(),
    )
    end_time = datetime.now()

    # ====================
    # Save run record
    # ====================
    record_data = {
        "pipeline_version": "2.0",  # 2.0: 5-stage breakdown (STT / LLM / MDM / coordinate-convert / 3DGS-gen) + pipeline-bus transport + phase/resource profiling
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "prompt": args.prompt,
        "scene": args.scene,
        "seed": args.seed,
        "steps": args.steps,
        "paths": {
            "run_dir": str(run_dir),
            "mdm_out": str(mdm_out_dir),
            "original_npz": str(target_npz) if 'target_npz' in locals() else None,
            "rotated_npz": str(rotated_npz),
            "final_mp4": str(final_mp4) if final_mp4 else None,
            "final_ply_dir": str(final_ply_dir) if 'final_ply_dir' in locals() and final_ply_dir else None,
        },
        "rotation": {
            "rx_degrees": 90,
            "rz_degrees": 180,
        },
        "translation": {
            "center": args.center,
            "tz_offset": args.tz,
            "ground": args.ground,
        },
        "repositories": {
            "mdm": str(args.mdm_repo),
            "hugs": str(args.hugs_repo),
        },
        "python_executables": {
            "mdm": str(args.mdm_py),
            "hugs": str(args.hugs_py),
        },
        "executed_commands": executed_commands,
        "scene_checkpoints": scene_cfg,
        "benchmark": {
            "total_duration_seconds": bench.total_seconds(),
            "stages": bench.stages,
        },
    }

    bench_json, bench_csv = bench.save(run_dir)

    record_path = run_dir / "run_record.json"
    with open(record_path, 'w') as f:
        json.dump(record_data, f, indent=2)
    print(f"\n✓ Saved run record to:  {record_path}")
    print(f"✓ Benchmark JSON:       {bench_json}")
    print(f"✓ Benchmark CSV:        {bench_csv}")
    
    # ====================
    # Summary
    # ====================
    print(f"\n{'='*80}")
    print("✓ Pipeline completed successfully!")
    print(f"{'='*80}")
    print(f"Duration:       {record_data['duration_seconds']:.1f} seconds")
    print(f"Run directory:  {run_dir}")
    print(f"Original npz:   {target_npz if 'target_npz' in locals() else 'N/A'}")
    print(f"Rotated npz:    {rotated_npz}")
    print(f"Final video:    {final_mp4 if final_mp4 else 'N/A'}")
    print(f"Posed PLYs:     {final_ply_dir if 'final_ply_dir' in locals() and final_ply_dir else 'N/A'}")
    print(f"Run record:     {record_path}")

    NAME_W = 22
    print(f"\n{'─'*90}")
    print("  Benchmark Timing  (CPU/RSS/GPU are averages sampled during each stage's subprocess)")
    print(f"{'─'*90}")
    header = f"  {'#':<4} {'Stage':<{NAME_W}} {'Duration':>9}  {'CPU%':>6} {'RSS(MB)':>8} {'GPU%':>6} {'VRAM(MB)':>9}  {'Status'}"
    print(header)
    print(f"  {'─'*4} {'─'*NAME_W} {'─'*9}  {'─'*6} {'─'*8} {'─'*6} {'─'*9}  {'─'*8}")
    for s in bench.stages:
        dur = f"{s['duration_seconds']:.1f}s" if s['duration_seconds'] is not None else '—'
        status_icon = {'success': '✓', 'failed': '✗', 'skipped': '–'}.get(s['status'], s['status'])
        res = s.get('resource') or {}
        cpu = f"{res['cpu_pct_avg']:.0f}" if res.get('cpu_pct_avg') is not None else '—'
        rss = f"{res['rss_mb_max']:.0f}" if res.get('rss_mb_max') is not None else '—'
        gpu = f"{res['gpu_util_pct_avg']:.0f}" if res.get('gpu_util_pct_avg') is not None else '—'
        vram = f"{res['gpu_mem_mb_max']:.0f}" if res.get('gpu_mem_mb_max') is not None else '—'
        print(f"  {s['stage']:<4} {s['name']:<{NAME_W}} {dur:>9}  {cpu:>6} {rss:>8} {gpu:>6} {vram:>9}  {status_icon} {s['status']}")
        phases = s.get('phases') or {}
        if phases:
            phase_str = "  ".join(f"{name}: {secs:.2f}s" for name, secs in phases.items())
            print(f"  {'':4} {'  ' + phase_str}")
    print(f"  {'─'*4} {'─'*NAME_W} {'─'*9}  {'─'*6} {'─'*8} {'─'*6} {'─'*9}  {'─'*8}")
    print(f"  {'':4} {'TOTAL':<{NAME_W}} {bench.total_seconds():.1f}s")
    print(f"{'─'*90}")
    print(f"  Benchmark CSV:  {bench_csv}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
