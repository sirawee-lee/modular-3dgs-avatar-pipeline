import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

k_all   = [1, 2, 4]
k_human = [1, 2]          # human-only data only exists for k=1,2

size_scene  = [12.2, 6.1, 3.1]
size_human  = [0.5,  0.25]

bw_scene    = [16.2, 8.1, 4.1]
bw_human    = [0.7,  0.3]

BLUE   = "#1f77b4"
ORANGE = "#e67e22"
GRAY   = "#555555"

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 4.6), sharey=False)

# ── left: animation size ──────────────────────────────────────────────────────
ax_l.plot(k_all,   size_scene, "o-",  color=BLUE,   linewidth=2.2, markersize=7,
          label="Scene + human")
ax_l.plot(k_human, size_human, "s--", color=ORANGE, linewidth=2.2, markersize=7,
          label="Human-only", zorder=5)

for x, y in zip(k_all, size_scene):
    ax_l.annotate(f"{y}", (x, y), textcoords="offset points",
                  xytext=(0, 8), ha="center", fontsize=9, color=BLUE)
for x, y in zip(k_human, size_human):
    ax_l.annotate(f"{y}", (x, y), textcoords="offset points",
                  xytext=(0, 8), ha="center", fontsize=9, color=ORANGE)

ax_l.set_xlabel("Subsampling factor $k$", fontsize=11)
ax_l.set_ylabel("Total animation size (GB)", fontsize=11)
ax_l.set_xticks(k_all)
ax_l.set_xlim(0.7, 4.5)
ax_l.set_ylim(-0.5, 14)
ax_l.yaxis.grid(True, linestyle="--", alpha=0.4)
ax_l.set_axisbelow(True)
ax_l.spines["top"].set_visible(False)
ax_l.spines["right"].set_visible(False)
ax_l.legend(fontsize=10, framealpha=0.9)
ax_l.set_title("Animation size", fontsize=12, pad=8)

# ── right: streaming bandwidth ────────────────────────────────────────────────
ax_r.plot(k_all,   bw_scene,  "o-",  color=BLUE,   linewidth=2.2, markersize=7,
          label="Scene + human")
ax_r.plot(k_human, bw_human,  "s--", color=ORANGE, linewidth=2.2, markersize=7,
          label="Human-only", zorder=5)

for x, y in zip(k_all, bw_scene):
    ax_r.annotate(f"{y}", (x, y), textcoords="offset points",
                  xytext=(0, 8), ha="center", fontsize=9, color=BLUE)
for x, y in zip(k_human, bw_human):
    ax_r.annotate(f"{y}", (x, y), textcoords="offset points",
                  xytext=(0, -14), ha="center", fontsize=9, color=ORANGE)

# Wi-Fi 6 reference line
ax_r.axhline(1.0, color=GRAY, linewidth=1.4, linestyle="--", zorder=2)
ax_r.text(4.42, 1.05, "Wi-Fi 6\nlimit", ha="right", va="bottom",
          fontsize=8.5, color=GRAY, linespacing=1.3)

ax_r.set_xlabel("Subsampling factor $k$", fontsize=11)
ax_r.set_ylabel("Required streaming bandwidth (Gbps)", fontsize=11)
ax_r.set_xticks(k_all)
ax_r.set_xlim(0.7, 4.5)
ax_r.set_ylim(-0.5, 19)
ax_r.yaxis.grid(True, linestyle="--", alpha=0.4)
ax_r.set_axisbelow(True)
ax_r.spines["top"].set_visible(False)
ax_r.spines["right"].set_visible(False)
ax_r.legend(fontsize=10, framealpha=0.9)
ax_r.set_title("Streaming bandwidth", fontsize=12, pad=8)

caption = (
    "Fig. 3 — System parameter: subsampling factor $k$ and rendering mode.\n"
    "Human-only mode is the dominant reduction; subsampling alone never crosses the Wi-Fi 6 threshold."
)
fig.text(0.5, -0.04, caption, ha="center", fontsize=9.5, style="italic",
         color="#333333")

plt.tight_layout()
plt.savefig("subsampling_chart.png", dpi=150, bbox_inches="tight")
print("Saved: subsampling_chart.png")
