import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── style (mirrors ImmerEdge25-plotting/gaussian_timing.ipynb) ──────────────
color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

plt.rcParams.update({
    'font.family':     'Liberation Serif',
    'font.size':       23,
    'axes.titlesize':  23,
    'axes.labelsize':  23,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'text.usetex':     False,
})

err_lw       = 1.5
err_capsize  = 4
err_capthick = 1.5
figsize = (8, 6)
# ────────────────────────────────────────────────────────────────────────────

stages = [
    "MDM\nMotion\nGeneration",
    "SMPL\nParameters\nExtraction",
    "Coordinate\nRotation",
    "HUGS\nRendering",
    "Video &\nPLY\nExtraction",
    "Total\nPipeline",
]

avg_times = [13.5936, 49.6118, 0.154, 369.6228, 11.2114, 444.1936]

# Assign a distinct color per bar; total bar gets gray
colors = color_palette[:5] + ['#7f7f7f']

fig, ax = plt.subplots(figsize=figsize)

x = np.arange(len(stages))
bars = ax.bar(x, avg_times, color=colors, width=0.6, zorder=3)

# Legend entries matching the image style: "Stage Name: value s"
legend_labels = [f"{name.replace(chr(10), ' ')}: {v:.1f}s"
                 for name, v in zip(stages, avg_times)]
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
ax.legend(handles, legend_labels,
          loc='upper left', fontsize=13, framealpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels(stages, fontsize=13)
ax.set_ylabel('Time (seconds)')
ax.set_ylim(0, max(avg_times) * 1.15)

ax.xaxis.grid(False)
ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

out = Path(__file__).parent.parent / 'pipeline_latency_v2.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
