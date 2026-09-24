#!/usr/bin/env python3
"""
Lightweight per-stage profiling for the text->HUGS pipeline: read/compute/write
sub-phase timing (PhaseTimer) and background CPU/memory/disk-IO/GPU resource
sampling (ResourceSampler). Both merge into scripts/run_text2hugs.py's
StageBenchmark / benchmark_timing.json rather than producing a separate report.

"%" note: CPU and GPU figures below are true utilization percentages. There's
no equivalent single "% busy" figure for disk I/O without deeper OS
instrumentation (iostat-style tracking of /proc/diskstats per device), so disk
is reported as throughput (MB read/written, delta over the sampled window)
instead. Memory is reported both as RSS in MB (the sampled process tree) and
as a percentage of total system RAM (psutil.virtual_memory().percent, sampled
concurrently, system-wide).
"""

import contextlib
import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

try:
    import psutil
except ImportError:
    psutil = None


class PhaseTimer:
    """Accumulates named phases (e.g. 'read' / 'compute' / 'write') across
    repeated calls -- durations SUM, they don't overwrite. This matters for
    loops (HUGS's per-frame animate() loop) where each phase recurs every
    iteration and we want the total time spent in each across the whole run.
    """

    def __init__(self) -> None:
        self.phases: Dict[str, float] = {}

    @contextlib.contextmanager
    def phase(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.phases[name] = self.phases.get(name, 0.0) + (time.perf_counter() - t0)

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.phases, indent=2))

    @staticmethod
    def load(path) -> Optional[Dict[str, float]]:
        path = Path(path)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None


class ResourceSampler:
    """Background-thread sampler attributing CPU%/RSS/mem%/disk-IO (psutil)
    and GPU util%/VRAM (nvidia-smi) to a target pid + its children (CUDA work
    often forks worker processes). Call start(pid) right after Popen returns,
    stop() once the process has exited, then summary() for the aggregated
    stats to fold into StageBenchmark.

    None-safe throughout: if psutil isn't installed, or there's no GPU /
    nvidia-smi on this box, the corresponding fields just stay None rather
    than raising.
    """

    def __init__(self, interval_hz: float = 8.0, gpu_index: int = 0) -> None:
        self.interval = 1.0 / interval_hz
        self.gpu_index = gpu_index
        self._samples: List[dict] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._nvsmi_proc: Optional[subprocess.Popen] = None

    def start(self, pid: int) -> None:
        self._samples = []
        self._stop.clear()
        self._nvsmi_proc = self._spawn_nvsmi()
        self._thread = threading.Thread(target=self._loop, args=(pid,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        if self._nvsmi_proc is not None:
            self._nvsmi_proc.terminate()
            try:
                self._nvsmi_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._nvsmi_proc.kill()
            self._nvsmi_proc = None

    def _spawn_nvsmi(self) -> Optional[subprocess.Popen]:
        # One long-lived --loop-ms process, read one line per tick, rather
        # than spawning nvidia-smi per sample -- spawn overhead alone would
        # exceed an 8Hz interval.
        try:
            return subprocess.Popen(
                [
                    "nvidia-smi",
                    "-i", str(self.gpu_index),
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                    f"--loop-ms={max(1, int(self.interval * 1000))}",
                ],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
        except (FileNotFoundError, OSError):
            return None

    def _read_nvsmi_line(self):
        if self._nvsmi_proc is None or self._nvsmi_proc.stdout is None:
            return None, None
        line = self._nvsmi_proc.stdout.readline()
        if not line:
            return None, None
        try:
            util_str, mem_str = line.strip().split(",")
            return float(util_str.strip()), float(mem_str.strip())
        except ValueError:
            return None, None

    def _loop(self, pid: int) -> None:
        proc = None
        if psutil is not None:
            try:
                proc = psutil.Process(pid)
                proc.cpu_percent(None)  # psutil gotcha: first call only sets a baseline
            except psutil.NoSuchProcess:
                proc = None

        while not self._stop.is_set():
            t0 = time.monotonic()
            sample = {"t": time.time()}

            if proc is not None:
                try:
                    procs = [p for p in [proc] + proc.children(recursive=True) if p.is_running()]
                    sample["cpu_pct"] = sum(p.cpu_percent(None) for p in procs)
                    sample["rss_mb"] = sum(p.memory_info().rss for p in procs) / 1e6

                    io_read_mb = io_write_mb = 0.0
                    have_io = False
                    for p in procs:
                        try:
                            io = p.io_counters()
                        except (psutil.AccessDenied, NotImplementedError, AttributeError):
                            continue
                        io_read_mb += io.read_bytes / 1e6
                        io_write_mb += io.write_bytes / 1e6
                        have_io = True
                    if have_io:
                        sample["io_read_mb"] = io_read_mb
                        sample["io_write_mb"] = io_write_mb
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                try:
                    sample["mem_pct"] = psutil.virtual_memory().percent
                except Exception:
                    pass

            gpu_util, gpu_mem = self._read_nvsmi_line()
            if gpu_util is not None:
                sample["gpu_util_pct"] = gpu_util
                sample["gpu_mem_mb"] = gpu_mem

            self._samples.append(sample)
            time.sleep(max(0.0, self.interval - (time.monotonic() - t0)))

    def summary(self) -> dict:
        def _avg_max(key):
            values = [s[key] for s in self._samples if key in s]
            if not values:
                return None, None
            return sum(values) / len(values), max(values)

        def _delta(key):
            values = [s[key] for s in self._samples if key in s]
            if not values:
                return None
            return max(values) - min(values)

        cpu_avg, cpu_max = _avg_max("cpu_pct")
        _, rss_max = _avg_max("rss_mb")
        mem_avg, _ = _avg_max("mem_pct")
        gpu_avg, gpu_max = _avg_max("gpu_util_pct")
        _, gpu_mem_max = _avg_max("gpu_mem_mb")

        return {
            "sample_count": len(self._samples),
            "cpu_pct_avg": cpu_avg, "cpu_pct_max": cpu_max,
            "rss_mb_max": rss_max,
            "mem_pct_avg": mem_avg,
            "io_read_mb": _delta("io_read_mb"), "io_write_mb": _delta("io_write_mb"),
            "gpu_util_pct_avg": gpu_avg, "gpu_util_pct_max": gpu_max,
            "gpu_mem_mb_max": gpu_mem_max,
        }


def merge_phase_dicts(phase_dicts) -> Optional[Dict[str, float]]:
    """Sum matching phase keys across several PhaseTimer.load() results --
    for a logical pipeline stage that's actually implemented as more than one
    subprocess (e.g. "coordinate converter" = SMPL fit + rotate), so the
    combined stage still reports one read/compute/write breakdown rather than
    silently keeping only the last sub-step's numbers. None/empty entries are
    skipped; returns None if every entry was None/empty."""
    merged: Dict[str, float] = {}
    for d in phase_dicts:
        if not d:
            continue
        for k, v in d.items():
            merged[k] = merged.get(k, 0.0) + v
    return merged or None


def merge_resource_summaries(summaries) -> dict:
    """Combine several ResourceSampler.summary() dicts (sub-steps of one
    logical stage, each its own subprocess) into one: CPU%/mem%/GPU% become
    sample-count-weighted averages, *_max fields take the max across inputs,
    disk-IO and sample_count sum. None-safe -- a field missing from every
    input just stays None in the result."""
    summaries = [s for s in summaries if s]
    if not summaries:
        return {}

    def _weighted_avg(key):
        total_w = 0
        total = 0.0
        for s in summaries:
            w = s.get('sample_count') or 0
            v = s.get(key)
            if v is None or w == 0:
                continue
            total += v * w
            total_w += w
        return (total / total_w) if total_w else None

    def _max(key):
        values = [s[key] for s in summaries if s.get(key) is not None]
        return max(values) if values else None

    def _sum(key):
        values = [s[key] for s in summaries if s.get(key) is not None]
        return sum(values) if values else None

    return {
        "sample_count": sum(s.get('sample_count') or 0 for s in summaries),
        "cpu_pct_avg": _weighted_avg('cpu_pct_avg'), "cpu_pct_max": _max('cpu_pct_max'),
        "rss_mb_max": _max('rss_mb_max'),
        "mem_pct_avg": _weighted_avg('mem_pct_avg'),
        "io_read_mb": _sum('io_read_mb'), "io_write_mb": _sum('io_write_mb'),
        "gpu_util_pct_avg": _weighted_avg('gpu_util_pct_avg'), "gpu_util_pct_max": _max('gpu_util_pct_max'),
        "gpu_mem_mb_max": _max('gpu_mem_mb_max'),
    }
