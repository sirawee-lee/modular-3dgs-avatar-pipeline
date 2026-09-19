#!/usr/bin/env python3
"""
Streaming cost vs. playback FPS — error-bar chart across 9 test motions.

Fill in `render_time_s` with one row per motion once measurements are collected.
Bandwidth is computed deterministically from fps and per-frame PLY size.
Saves both PNG and EPS (Type-42 fonts, no Type-3 bitmaps).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

plt.rcParams.update({
    'font.family':     'Liberation Serif',
    'font.size':       13,
    'axes.labelsize':  13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'text.usetex':     False,
    'ps.fonttype':     42,
})

# ── constants ─────────────────────────────────────────────────────────────────
FPS_BASE = 20          # original animation fps
FILE_MB  = 101.7       # per-frame PLY size (MB), human-only export

k_values  = [1,  2,  4,   8,    16]
fps_labels = ['20\n(k=1)', '10\n(k=2)', '5\n(k=4)', '2.5\n(k=8)', '1.25\n(k=16)']
fps_values = [FPS_BASE / k for k in k_values]   # [20, 10, 5, 2.5, 1.25]

# Deterministic: fps × file_size(MB) × 8 bits / 1000 → Gbps
bw_gbps = np.array([fps * FILE_MB * 8 / 1000 for fps in fps_values])

# ── render time measurements (seconds) ───────────────────────────────────────
# Rows = motions (9 total), columns = k values [k=1, k=2, k=4, k=8, k=16]
# Replace np.nan with measured values as you collect them.
render_time_s = np.array([
    # k=1     k=2     k=4     k=8     k=16
    [370.5,  215.0,  124.3,  np.nan, np.nan],  # motion 1
    [np.nan, np.nan, np.nan, np.nan, np.nan],  # motion 2
    [np.nan, np.nan, np.nan, np.nan, np.nan],  # motion 3
    [np.nan, np.nan, np.nan, np.nan, np.nan],  # motion 4
    [np.nan, np.nan, np.nan, np.nan, np.nan],  # motion 5
    [np.nan, np.nan, np.nan, np.nan, np.nan],  # motion 6
    [np.nan, np.nan, np.nan, np.nan, np.nan],  # motion 7
    [np.nan, np.nan, np.nan, np.nan, np.nan],  # motion 8
    [np.nan, np.nan, np.nan, np.nan, np.nan],  # motion 9
], dtype=float)

# ── statistics (95% CI) ───────────────────────────────────────────────────────
def mean_ci95(col):
    data = col[~np.isnan(col)]
    n = len(data)
    if n == 0:
        return np.nan, np.nan
    m = np.nanmean(data)
    if n == 1:
        return m, 0.0   # single point: no CI
    se = stats.sem(data)
    ci = se * stats.t.ppf(0.975, df=n - 1)
    return m, ci

rt_mean = np.array([mean_ci95(render_time_s[:, j])[0] for j in range(len(k_values))])
rt_ci   = np.array([mean_ci95(render_time_s[:, j])[1] for j in range(len(k_values))])

# ── plot ──────────────────────────────────────────────────────────────────────
BLUE   = '#1f77b4'
ORANGE = '#ff7f0e'
GRAY   = '#555555'

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 5))
x = np.arange(len(k_values))

# left panel — render time
valid = ~np.isnan(rt_mean)
ax_l.errorbar(x[valid], rt_mean[valid], yerr=rt_ci[valid],
              fmt='o-', color=BLUE, linewidth=2.2, markersize=7,
              capsize=5, capthick=1.5, elinewidth=1.5, label='Render time (mean ± 95% CI)')
ax_l.set_xticks(x)
ax_l.set_xticklabels(fps_labels)
ax_l.set_xlabel('Playback FPS')
ax_l.set_ylabel('Render time (s)')
ax_l.set_ylim(bottom=0)
ax_l.yaxis.grid(True, linestyle='--', alpha=0.4)
ax_l.set_axisbelow(True)
ax_l.spines['top'].set_visible(False)
ax_l.spines['right'].set_visible(False)
ax_l.set_title('Render time vs. playback FPS', pad=8)

# right panel — bandwidth
ax_r.plot(x, bw_gbps, 's-', color=ORANGE, linewidth=2.2, markersize=7,
          label='Required bandwidth')
for xi, bw in zip(x, bw_gbps):
    ax_r.annotate(f'{bw:.2f}', (xi, bw),
                  textcoords='offset points', xytext=(0, 9),
                  ha='center', fontsize=9, color=ORANGE)

# Wi-Fi 6 reference line
ax_r.axhline(1.0, color=GRAY, linewidth=1.5, linestyle='--', zorder=2)
ax_r.text(len(k_values) - 0.55, 1.08, 'Wi-Fi 6 limit\n(~1 Gbps)',
          ha='right', va='bottom', fontsize=9, color=GRAY, linespacing=1.3)

ax_r.set_xticks(x)
ax_r.set_xticklabels(fps_labels)
ax_r.set_xlabel('Playback FPS')
ax_r.set_ylabel('Required streaming bandwidth (Gbps)')
ax_r.set_ylim(bottom=0)
ax_r.yaxis.grid(True, linestyle='--', alpha=0.4)
ax_r.set_axisbelow(True)
ax_r.spines['top'].set_visible(False)
ax_r.spines['right'].set_visible(False)
ax_r.set_title('Bandwidth vs. playback FPS', pad=8)

plt.tight_layout()

out_dir = Path(__file__).parent.parent
plt.savefig(out_dir / 'streaming_cost.png', dpi=150, bbox_inches='tight')
plt.savefig(out_dir / 'streaming_cost.eps', dpi=150, bbox_inches='tight')
print(f'Saved to {out_dir}/')
